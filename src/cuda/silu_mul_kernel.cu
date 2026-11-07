#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
__global__ void silu_mul_kernel(half* out, const half* gate, const half* up, const int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        float gate_val = __half2float(gate[idx]);
        float up_val = __half2float(up[idx]);
        out[idx] = __float2half(gate_val / (1.0f + expf(-gate_val)) * up_val);
    }
}

// 2. The Host Wrapper (Called by C++)
torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.device().is_cuda(), "Tensor gate must be on a CUDA device");
    TORCH_CHECK(up.device().is_cuda(), "Tensor up must be on a CUDA device");
    TORCH_CHECK(gate.is_contiguous(), "Tensor gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "Tensor up must be contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    TORCH_CHECK(gate.scalar_type() == at::kHalf && up.scalar_type() == at::kHalf, "gate and up must be fp16");

    auto c = torch::empty_like(gate);
    int elements = gate.numel();
    int threads = 256;
    int blocks = (elements + threads - 1) / threads;

    silu_mul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}
