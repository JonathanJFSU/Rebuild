import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Create random matrices
matrix_size = 25000
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Perform matrix multiplication on GPU
result = torch.matmul(a, b)

# Optional: Move result back to CPU if needed
result_cpu = result.to("cpu")

print("Matrix multiplication complete. Result shape:", result_cpu.shape)
