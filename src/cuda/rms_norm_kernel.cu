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
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.device().is_cuda(), "Tensor x must be on a CUDA device");
    TORCH_CHECK(weight.device().is_cuda(), "Tensor weight must be on a CUDA device");
    TORCH_CHECK(weight.is_contiguous(), "Tensor weight must be contiguous");

    auto size = x.numel();
    auto c = torch::empty_like(x);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        c.data_ptr<float>(),
        size
    );

    return c;
        c.data_ptr<float>(),
        size
    );

    return c;
}
