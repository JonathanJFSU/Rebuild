
import cv2
import torch
from transformers import AutoProcessor, AutoModel
import numpy as np
from collections import Counter
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


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

def rotate_frame(frame):
    """Rotate frame 90 degrees clockwise."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

def extract_frames(video_path):
    """Extract all frames from video and rotate them 90 degrees clockwise."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    with tqdm(total=total_frames, desc="Extracting and rotating frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
               
            # Rotate frame 90 degrees clockwise
            frame = rotate_frame(frame)
           
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            pbar.update(1)
   
    cap.release()
    return frames

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

def process_frames(frames, yolo_model, siglip_processor, siglip_model, object_labels, action_labels, output_video_path=None):
    """Process frames using YOLO pose estimation and SigLIP."""
    all_object_predictions = []
    all_action_predictions = []
    all_shirt_colors = []
    all_movement_directions = []
    prev_keypoints = None
    processed_frames = []
   
    # Set up video writer if output path is provided
    if output_video_path:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
   
    with tqdm(total=len(frames), desc="Processing frames") as pbar:
        for frame in frames:
            # Run YOLO pose estimation
            results = yolo_model(frame, verbose=False)
            keypoints = None
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints
           
            # Process with SigLIP
            inputs_objects = siglip_processor(images=frame, text=object_labels, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs_objects = siglip_model(**inputs_objects)
                logits_objects = outputs_objects.logits_per_image
                probs_objects = logits_objects.softmax(dim=1)
                top_obj_idx = probs_objects[0].argmax().item()
                predicted_object = object_labels[top_obj_idx]
                all_object_predictions.append(predicted_object)

                direction = "unknown"
                shirt_color = None
                if predicted_object == "person" and keypoints is not None:
                    # Get shirt color using pose keypoints
                    shirt_color = detect_shirt_color(frame, keypoints)
                    if shirt_color:
                        all_shirt_colors.append(shirt_color)
                   
                    # Get movement direction using keypoints
                    direction = get_movement_direction(prev_keypoints, keypoints)
                    if direction != "unknown":
                        all_movement_directions.append(direction)
                    prev_keypoints = keypoints

                    # Process actions with context from pose and movement
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
                    ] + action_labels  # Add original action labels as fallback

                    inputs_actions = siglip_processor(images=frame, text=action_context, return_tensors="pt", padding=True)
                    outputs_actions = siglip_model(**inputs_actions)
                    logits_actions = outputs_actions.logits_per_image
                    probs_actions = logits_actions.softmax(dim=1)
                   
                    # Get top 3 actions and their probabilities
                    top_3_probs, top_3_indices = torch.topk(probs_actions[0], 3)
                   
                    # Only accept prediction if confidence is high enough
                    if top_3_probs[0] > 0.3:  # Confidence threshold
                        predicted_action = action_context[top_3_indices[0]]
                    else:
                        # Fall back to movement-based action if confidence is low
                        if "moving" in direction:
                            predicted_action = "walking"
                        elif "standing still" in direction:
                            predicted_action = "standing"
                        else:
                            predicted_action = "moving"
                else:
                    # If no person detected, process general actions
                    inputs_actions = siglip_processor(images=frame, text=action_labels, return_tensors="pt", padding=True)
                    outputs_actions = siglip_model(**inputs_actions)
                    logits_actions = outputs_actions.logits_per_image
                    probs_actions = logits_actions.softmax(dim=1)
                    top_action_idx = probs_actions[0].argmax().item()
                    predicted_action = action_labels[top_action_idx]

                all_action_predictions.append(predicted_action)
           
            # Create visualization frame
            if output_video_path:
                vis_frame = create_visualization(frame, results, predicted_object,
                                              predicted_action, direction, shirt_color)
                out.write(vis_frame)
           
            pbar.update(1)
   
    if output_video_path:
        out.release()
   
    return all_object_predictions, all_action_predictions, all_shirt_colors, all_movement_directions

def main():
    parser = argparse.ArgumentParser(description='Process video with YOLO pose estimation and SigLIP')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_path', type=str, help='Path to save the results text file')
    parser.add_argument('--output_video', type=str, default=None,
                      help='Path to save the output video with detections')
    args = parser.parse_args()

    # Initialize models
    print("Loading models...")
    yolo_model = YOLO('yolov8n-pose.pt')  # Fixed model name
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")


    # Define labels for SigLIP
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

    # Extract frames
    print("Extracting frames from video...")
    frames = extract_frames(args.video_path)

    # Process frames
    print("Processing frames with YOLO and SigLIP...")
    object_predictions, action_predictions, shirt_colors, movement_directions = process_frames(
        frames, yolo_model, siglip_processor, siglip_model, object_labels, action_labels,
        args.output_video
    )

    # Get top 5 most common predictions
    object_counter = Counter(object_predictions)
    action_counter = Counter(action_predictions)
    shirt_color_counter = Counter(shirt_colors)
    direction_counter = Counter(movement_directions)
    top_5_objects = object_counter.most_common(5)
    top_5_actions = action_counter.most_common(5)
    top_5_colors = shirt_color_counter.most_common(5)
    top_5_directions = direction_counter.most_common(5)

    # Calculate person detection statistics
    person_frames = object_predictions.count("person")
    total_frames = len(object_predictions)
    person_percentage = (person_frames / total_frames) * 100 if total_frames > 0 else 0

    # Save results
    with open(args.output_path, 'w') as f:
        f.write("Top 5 objects identified in the video:\n")
        f.write("-" * 40 + "\n")
        for item, count in top_5_objects:
            percentage = (count / len(object_predictions)) * 100
            f.write(f"{item}: {percentage:.2f}% ({count} frames)\n")
       
        f.write("\nTop 5 actions identified in the video:\n")
        f.write("-" * 40 + "\n")
        for item, count in top_5_actions:
            percentage = (count / len(action_predictions)) * 100
            f.write(f"{item}: {percentage:.2f}% ({count} frames)\n")

        if person_frames > 0:
            f.write(f"\nPerson detected in {person_percentage:.2f}% of frames ({person_frames} frames)\n")
           
            f.write("\nTop 5 movement directions (from pose estimation):\n")
            f.write("-" * 40 + "\n")
            for direction, count in top_5_directions:
                percentage = (count / person_frames) * 100
                f.write(f"{direction}: {percentage:.2f}% ({count} frames)\n")
           
            f.write("\nTop 5 shirt colors detected (hex values):\n")
            f.write("-" * 40 + "\n")
            for color, count in top_5_colors:
                percentage = (count / person_frames) * 100
                f.write(f"{color}: {percentage:.2f}% ({count} frames)\n")

    print(f"Results have been saved to {args.output_path}")
    if args.output_video:
        print(f"Processed video has been saved to {args.output_video}")

if __name__ == "__main__":
    main()