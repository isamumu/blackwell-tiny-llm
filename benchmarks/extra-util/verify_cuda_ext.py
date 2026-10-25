import torch
from torch.utils.cpp_extension import load
import os
import sys

print("Compiling and loading custom CUDA extension from benchmarks folder...")

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/usr/local/cuda"

# Updated paths using '../' to step out of benchmarks/ and into src/
cuda_extension = load(
    name="cuda_vector_extension",
    sources=["../src/cuda/vector_add.cpp", "../src/cuda/vector_add_kernel.cu"],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-std=c++17"],
    verbose=True
)

print("Extension loaded successfully!\n")

if __name__ == "__main__":
    # 1. Check if PyTorch sees your GPU
    assert torch.cuda.is_available(), "CUDA is not available to PyTorch!"
    
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Active Verification Device: [{current_device}] {device_name}")

    size = 50_000_000  # Bumped up to 50 Million elements to give you time to track it
    tensor_a = torch.randn(size, device="cuda", dtype=torch.float32)
    tensor_b = torch.randn(size, device="cuda", dtype=torch.float32)

    # 2. Assert tensors are explicitly allocated on the GPU memory graph
    print(f"Tensor A Device allocation: {tensor_a.device} (Expecting cuda:0)")
    print(f"Tensor B Device allocation: {tensor_b.device} (Expecting cuda:0)")

    # 3. Synchronize device before and after launching the custom kernel
    torch.cuda.synchronize() 
    print("Launching kernel on GPU...")
    
    custom_result = cuda_extension.vector_add(tensor_a, tensor_b)
    
    torch.cuda.synchronize() # Forces the CPU to wait for the GPU to finish computation
    print("Kernel execution completed on hardware.")
