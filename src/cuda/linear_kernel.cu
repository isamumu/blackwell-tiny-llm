#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
__global__ void linear_kernel(half* out, const half* x, const half* weight,
                              const int rows, const int in_features, const int out_features) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // global warp number → n
    int lane    = threadIdx.x % 32;                                // 0..31 within the warp

    if (warp_id >= out_features) return;

    
    const half* w_row = weight + warp_id * in_features;
    float sum = 0.0f;
    for (int k = lane; k < in_features; k += 32)
        sum += __half2float(x[k]) * __half2float(w_row[k]);

    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0)
        out[warp_id] = __float2half(sum);

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

    const int rows = x.numel() / in_features;
    auto out = torch::empty({rows, out_features}, x.options());
    if (rows != 1) return at::linear(x, weight);      // GEMV only; prefill → PyTorch

    int threads = 256;                                 // 8 warps per block
    int warps_per_block = threads / 32;
    int blocks = (out_features + warps_per_block - 1) / warps_per_block;

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
