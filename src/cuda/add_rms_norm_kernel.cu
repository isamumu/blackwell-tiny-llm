#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
// new_residual = residual + input, out = rms_norm(new_residual) * weight
__global__ void add_rms_norm_kernel(half* out, half* residual_out, const half* input, const half* residual, const half* weight, const float epsilon, const int hidden_dim) {
    // one block per row, threads do element slicing
    const half* input_row = input + blockIdx.x * hidden_dim;
    const half* residual_row = residual + blockIdx.x * hidden_dim;
    half* out_row = out + blockIdx.x * hidden_dim;
    half* residual_out_row = residual_out + blockIdx.x * hidden_dim;

    // one partial sum per thread (blockDim.x <= 1024)
    __shared__ float shared_mem[1024];

    // residual add + sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        // add in half (same rounding as the reference), then square the rounded value in fp32
        half sum = __hadd(input_row[i], residual_row[i]);
        residual_out_row[i] = sum;  // second output: the updated residual stream
        float val = __half2float(sum);
        local_sum += val * val;
    }
    shared_mem[threadIdx.x] = local_sum;
    __syncthreads();

    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // normalize: each thread re-reads only the elements it wrote above, so no extra sync is needed
    float rms = sqrtf(shared_mem[0] / hidden_dim + epsilon);
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        // same rounding order as the reference: normalize in fp32, cast to half, then scale by weight in half
        float val = __half2float(residual_out_row[i]);
        out_row[i] = __hmul(weight[i], __float2half(val / rms));
    }
}

// 2. The Host Wrapper (Called by C++)
std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.device().is_cuda(), "Tensor x must be on a CUDA device");
    TORCH_CHECK(residual.device().is_cuda(), "Tensor residual must be on a CUDA device");
    TORCH_CHECK(weight.device().is_cuda(), "Tensor weight must be on a CUDA device");
    TORCH_CHECK(weight.is_contiguous(), "Tensor weight must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && residual.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf, "x, residual, and weight must be fp16");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have the same shape");
    TORCH_CHECK(weight.numel() == x.size(-1), "weight must have hidden_dim elements");

    x = x.contiguous();  // rows must sit back-to-back for blockIdx.x * hidden_dim indexing
    residual = residual.contiguous();
    auto out = torch::empty_like(x);
    auto residual_out = torch::empty_like(residual);

    int hidden = x.size(-1);
    int rows = x.numel() / hidden;
    int threads = 256;  // must be a power of two for the tree reduction

    // one block per row, on PyTorch's current stream (required for CUDA graph capture later)
    add_rms_norm_kernel<<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(residual_out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        eps,
        hidden
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, residual_out};
}
