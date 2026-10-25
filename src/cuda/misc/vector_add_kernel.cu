#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 1. The Device Code (Runs on GPU threads)
__global__ void vector_add_kernel(const float* __restrict__ a, 
                                   const float* __restrict__ b, 
                                   float* __restrict__ c, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// 2. The Host Wrapper (Called by C++)
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on a CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on a CUDA device");
    TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor b must be contiguous");

    auto size = a.numel();
    auto c = torch::empty_like(a);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    return c;
}
