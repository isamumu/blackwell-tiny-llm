#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
__global__ void linear_kernel(half* out, const half* x, const half* weight,
                              const int rows, const int in_features, const int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_features) return;

    int row = idx / out_features; // out_features is how many rows there are for weights
    int col = idx % out_features;
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += __half2float(x[row * in_features + k]) *
               __half2float(weight[col * in_features + k]);
    }
    out[row * out_features + col] = __float2half(sum);

}

// 2. The Host Wrapper (Called by C++)
torch::Tensor linear_cuda(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.device().is_cuda(), "Tensor x must be on a CUDA device");
    TORCH_CHECK(weight.device().is_cuda(), "Tensor weight must be on a CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf, "x and weight must be fp16");
    TORCH_CHECK(x.dim() >= 2 && weight.dim() == 2, "x must be [..., in_features] and weight [out_features, in_features]");

    const int in_features = x.size(-1);
    const int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight must have shape [out_features, in_features]");

    x = x.contiguous();
    weight = weight.contiguous();

    auto out = torch::empty({x.numel() / in_features, out_features}, x.options());
    const int rows = x.numel() / in_features;
    int threads = 256;
    int blocks = (rows * out_features + threads - 1) / threads;

    linear_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        rows,
        in_features,
        out_features
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_sizes = x.sizes().vec();
    out_sizes.pop_back();
    out_sizes.push_back(out_features);
    return out.view(out_sizes);
}
