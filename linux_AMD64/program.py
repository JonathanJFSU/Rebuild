import cv2
import torch
import threading
from queue import Queue, Empty
import numpy as np
from collections import Counter
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel
import argparse
from tqdm import tqdm
import time

# ------------- SigLIP functions from paulfilereal.py ------------

def get_movement_direction(prev_keypoints, curr_keypoints):
    """Determine movement direction based on keypoint changes."""
    if prev_keypoints is None or curr_keypoints is None:
        return "unknown"

    try:
        # YOLO keypoints are in format [x, y, confidence]
        # Get mid point between shoulders (keypoints 5 and 6) for more stable tracking
        curr_left_shoulder = curr_keypoints.data[0][5][:2]  # Only take x,y coordinates
        curr_right_shoulder = curr_keypoints.data[0][6][:2]
        curr_center = [(curr_left_shoulder[0] + curr_right_shoulder[0])/2,
                       (curr_left_shoulder[1] + curr_right_shoulder[1])/2]

        prev_left_shoulder = prev_keypoints.data[0][5][:2]
        prev_right_shoulder = prev_keypoints.data[0][6][:2]
        prev_center = [(prev_left_shoulder[0] + prev_right_shoulder[0])/2,
                       (prev_left_shoulder[1] + prev_right_shoulder[1])/2]

        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]

        # Define thresholds for movement detection
        threshold = 5  # pixels - lowered for more sensitive detection

        # Determine primary direction
        if abs(dx) < threshold and abs(dy) < threshold:
            return "standing still"

        directions = []
        if abs(dx) > threshold:
            directions.append("right" if dx > 0 else "left")
        if abs(dy) > threshold:
            directions.append("down" if dy > 0 else "up")

        if not directions:
            return "standing still"
        elif len(directions) == 1:
            return f"moving {directions[0]}"
        else:
            return f"moving {directions[0]}-{directions[1]}"
    except (IndexError, AttributeError, TypeError) as e:
        return "unknown"


def rgb_to_hex(rgb):
    """Convert RGB color to hex value."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def detect_shirt_color(frame, keypoints):
    """Detect the dominant color in the upper body region using pose keypoints."""
    try:
        # Get coordinates for upper body region using keypoints
        left_shoulder = keypoints.data[0][5][:2]  # Only take x,y coordinates
        right_shoulder = keypoints.data[0][6][:2]
        left_hip = keypoints.data[0][11][:2]
        right_hip = keypoints.data[0][12][:2]

        # Calculate the region of interest
        top = int(min(left_shoulder[1], right_shoulder[1]))
        bottom = int(max(left_hip[1], right_hip[1]))
        left = int(min(left_shoulder[0], left_hip[0]))
        right = int(max(right_shoulder[0], right_hip[0]))

        # Ensure coordinates are within frame boundaries
        height, width = frame.shape[:2]
        top = max(0, top)
        bottom = min(height, bottom)
        left = max(0, left)
        right = min(width, right)

        # Extract upper body region
        upper_body = frame[top:bottom, left:right]
        if upper_body.size == 0:
            return None

        # Calculate the average color in the region
        average_color = np.mean(upper_body, axis=(0, 1))
        # Convert to hex color
        hex_color = rgb_to_hex(average_color)
        return hex_color
    except (IndexError, AttributeError, TypeError) as e:
        return None


def draw_text(img, text, position, scale=0.7, color=(255, 255, 255), thickness=2):
    """Helper function to draw text with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    # Draw background rectangle
    x, y = position
    cv2.rectangle(img, (x, y - text_size[1] - 10),
                  (x + text_size[0] + 10, y + 5), (0, 0, 0), -1)

    # Draw text
    cv2.putText(img, text, (x + 5, y), font, scale, color, thickness)
    return text_size[1] + 15  # Return height of text block


def create_visualization(frame, results, predicted_object, predicted_action, direction, shirt_color):
    """Create visualization with all detections and information."""
    # Convert frame to BGR for visualization
    vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw YOLO pose estimation results
    if results and len(results) > 0:
        annotated_frame = results[0].plot()
        vis_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Draw detection information
    y_offset = 30
    text_height = draw_text(vis_frame, f"Object: {predicted_object}", (10, y_offset))
    y_offset += text_height

    text_height = draw_text(vis_frame, f"Action: {predicted_action}", (10, y_offset))
    y_offset += text_height

    if direction != "unknown":
        text_height = draw_text(vis_frame, f"Movement: {direction}", (10, y_offset))
        y_offset += text_height

    if shirt_color:
        text_height = draw_text(vis_frame, f"Shirt color: {shirt_color}", (10, y_offset))

    return vis_frame


# ---------------------------------------------------------------

def yolo_worker(frame_queue, yolo_queue, yolo_model, stop_event):
    while not stop_event.is_set():
        try:
            idx, frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue
        with torch.no_grad():
            results = yolo_model(frame, verbose=False)
        yolo_queue.put((idx, frame, results))
        frame_queue.task_done()

def siglip_worker(yolo_queue, display_queue, siglip_processor, siglip_model, object_labels, action_labels, stop_event):
    prev_keypoints = None
    while not stop_event.is_set():
        try:
            idx, frame, yolo_results = yolo_queue.get(timeout=0.5)
        except Empty:
            continue

        keypoints = None
        if len(yolo_results) > 0 and yolo_results[0].keypoints is not None and len(yolo_results[0].keypoints) > 0:
            keypoints = yolo_results[0].keypoints

        # Object classification
        inputs_objects = siglip_processor(images=frame, text=object_labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_objects = siglip_model(**inputs_objects)
            logits_objects = outputs_objects.logits_per_image
            probs_objects = logits_objects.softmax(dim=1)
            top_obj_idx = probs_objects[0].argmax().item()
            predicted_object = object_labels[top_obj_idx]

            direction = "unknown"
            shirt_color = None
            if predicted_object == "person" and keypoints is not None:
                shirt_color = detect_shirt_color(frame, keypoints)
                direction = get_movement_direction(prev_keypoints, keypoints)
                prev_keypoints = keypoints

                # Action context for SigLIP
                action_context = [
                    f"person {direction}",  # Add movement direction context
                    "person walking normally",
                    "person walking casually",
                    "person strolling",
                    "person moving at normal pace",
                    "person taking steps",
                    "person walking forward",
                    "person walking in a straight line",
                    "person moving forward",
                    "person in motion",
                ] + action_labels

                inputs_actions = siglip_processor(images=frame, text=action_context, return_tensors="pt", padding=True)
                outputs_actions = siglip_model(**inputs_actions)
                logits_actions = outputs_actions.logits_per_image
                probs_actions = logits_actions.softmax(dim=1)

                # Top action
                top_3_probs, top_3_indices = torch.topk(probs_actions[0], 3)
                if top_3_probs[0] > 0.3:
                    predicted_action = action_context[top_3_indices[0]]
                else:
                    if "moving" in direction:
                        predicted_action = "walking"
                    elif "standing still" in direction:
                        predicted_action = "standing"
                    else:
                        predicted_action = "moving"
            else:
                inputs_actions = siglip_processor(images=frame, text=action_labels, return_tensors="pt", padding=True)
                outputs_actions = siglip_model(**inputs_actions)
                logits_actions = outputs_actions.logits_per_image
                probs_actions = logits_actions.softmax(dim=1)
                top_action_idx = probs_actions[0].argmax().item()
                predicted_action = action_labels[top_action_idx]

        vis_frame = create_visualization(frame, yolo_results, predicted_object, predicted_action, direction, shirt_color)
        # Add everything needed for stats here
        display_queue.put((idx, vis_frame, predicted_object, predicted_action, shirt_color, direction))
        yolo_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description="YOLO + SigLIP Multithreaded Video Analyzer")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=None, help='Output video with overlays')
    parser.add_argument('--output_txt', type=str, default="results.txt", help='Text output of statistics')
    parser.add_argument('--yolo_model', type=str, default='yolov8n-pose.pt')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()

    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO(args.yolo_model)
    print("Loading SigLIP model...")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

    # SigLIP labels
    object_labels = [
        "person", "car", "building", "tree", "sky", "road", "animal",
        "furniture", "electronics", "food", "clothing", "vehicle",
        "nature", "indoor scene", "outdoor scene"
    ]
    action_labels = [
        "walking", "running", "jogging", "standing", "sitting",
        "moving slowly", "moving quickly", "stopping",
        "turning", "walking forward", "walking backward",
        "standing still", "starting to walk", "continuing to walk",
        "moving at regular pace", "taking steps", "striding",
        "moving naturally", "proceeding forward", "advancing"
    ]

    # Set up video
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optional video writer
    out_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Queues and stop event
    frame_queue = Queue(maxsize=args.threads * 2)
    yolo_queue = Queue(maxsize=args.threads * 2)
    display_queue = Queue(maxsize=args.threads * 2)
    stop_event = threading.Event()

    # Start worker threads
    yolo_threads = [
        threading.Thread(target=yolo_worker, args=(frame_queue, yolo_queue, yolo_model, stop_event), daemon=True)
        for _ in range(args.threads)
    ]
    for t in yolo_threads: t.start()

    siglip_threads = [
        threading.Thread(target=siglip_worker, args=(yolo_queue, display_queue, siglip_processor, siglip_model, object_labels, action_labels, stop_event), daemon=True)
        for _ in range(args.threads)
    ]
    for t in siglip_threads: t.start()

    stats = {'objects': [], 'actions': [], 'colors': [], 'directions': []}
    idx = 0
    start_time = time.time()

    # Main loop: read, enqueue, display, and collect stats
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put((idx, frame_rgb))
            idx += 1

            # Display/collect as results are available
            try:
                while True:
                    result = display_queue.get_nowait()
                    ridx, vis_frame, predicted_object, predicted_action, shirt_color, direction = result
                    # Display (convert back to BGR for OpenCV)
                    show_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                    print("vis_frame shape:", vis_frame.shape, "show_frame shape:", show_frame.shape, "expected:", (height, width))
                    #cv2.imshow("YOLO + SigLIP Analysis", show_frame)
                    if out_writer: out_writer.write(show_frame)
                    if predicted_object: stats['objects'].append(predicted_object)
                    if predicted_action: stats['actions'].append(predicted_action)
                    if shirt_color: stats['colors'].append(shirt_color)
                    if direction and direction != "unknown": stats['directions'].append(direction)
                    pbar.update(1)
                    display_queue.task_done()
            except Empty:
                pass

            # Break if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    stop_event.set()
    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()

    # Wait for threads to finish
    for t in yolo_threads + siglip_threads:
        t.join(timeout=1.0)

    # Aggregate stats and save to file
    def top5(counter_list):
        return Counter(counter_list).most_common(5)

    with open(args.output_txt, "w") as f:
        f.write("Top 5 objects:\n")
        for obj, count in top5(stats['objects']):
            f.write(f"{obj}: {count}\n")
        f.write("\nTop 5 actions:\n")
        for act, count in top5(stats['actions']):
            f.write(f"{act}: {count}\n")
        f.write("\nTop 5 shirt colors (if detected):\n")
        for color, count in top5(stats['colors']):
            f.write(f"{color}: {count}\n")
        f.write("\nTop 5 movement directions (if detected):\n")
        for dirn, count in top5(stats['directions']):
            f.write(f"{dirn}: {count}\n")
        f.write("\nProcessed %d frames in %.2fs\n" % (idx, time.time() - start_time))

    print(f"Results written to {args.output_txt}")
    if args.output_video:
        print(f"Video written to {args.output_video}")

if __name__ == "__main__":
    main()
