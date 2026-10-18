# blackwell-tiny-llm


Phase 1: Establish the Python Baseline (Day 1)

Do not write a single line of CUDA yet. Write a minimal inference script using standard PyTorch eager mode.
• The Code: Download Llama-3.2-1B or Gemma-2-2B via Hugging Face. Write a raw Python class that handles the forward pass (Matrix Multiplications, Attention, RMSNorm, Rotary Embeddings).
• The Goal: Get text generation working. Create a benchmarking script using torch.cuda.Event to measure Tokens Per Second and memory consumption. This is your baseline.

Phase 2: Create the C++/CUDA Extension Harness (Day 2)

Set up the bridge between Python and your GPU using torch.utils.cpp_extension.
• The Code: Write a basic "Hello World" CUDA kernel that simply adds two vectors together. Bind it to Python using TORCH_EXTENSION_NAME.
• The Goal: Verify that you can call a custom C++ function in your Python model script and have it execute successfully on your RTX 5060.

Phase 3: Optimize the Low-Hanging Fruit (The "Fused" Kernels)

Look at your model profile. Operations like RMSNorm and activations (SilU/SwiGLU) are memory-bandwidth bound. They waste time reading and writing to VRAM.
• The Code: Write a fused CUDA kernel for RMSNorm + Bias Addition. Instead of multiple sequential PyTorch steps, your single kernel loads data into registers once, calculates the norm, adds bias, and writes it back.
• The Goal: Drop your custom kernel into your Phase 1 model script. Run your benchmark. You will see an immediate, measurable speedup in token generation.

Phase 4: Write Custom Tiled Matrix Multiplication (GEMM)

Now tackle the heaviest math. Replace standard linear layers with your own matrix multiplication kernel.
• The Code: Implement Shared Memory Tiling. Threads should cooperatively load chunks of weights and activations into fast __shared__ memory banks to maximize data reuse.
• The Goal: Keep optimizing this kernel using NVIDIA Nsight Compute to eliminate bank conflicts and improve compute utilization [On Thu, October 15, 2026 @ 01:13 PM].

Phase 5: Implement On-The-Fly Dequantization

To fit larger contexts or models comfortably on your 8GB card, compress the weights.
• The Code: Quantize your model weights to 4-bit or 8-bit integers offline. Write a CUDA kernel that reads these compressed integer weights from VRAM, unpacks/dequantizes them into half-precision values right inside the GPU registers, and immediately passes them to your tiled GEMM kernel.
