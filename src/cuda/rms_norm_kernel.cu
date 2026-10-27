#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
__global__ void rms_norm_kernel(half* out, const half* input, const half* weight, const float epsilon, const int hidden_dim) {
    // one block per row, threads do element slicing
    const half* input_row = input + blockIdx.x * hidden_dim;
    half* out_row = out + blockIdx.x * hidden_dim;

    // assume 1024 max threads per block, so we can use a single warp for reduction
    __shared__ float shared_mem[1024]; // btw 32 threads in a warp

    // sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(input_row[i]);
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

    // normalize
    float rms = sqrtf(shared_mem[0] / hidden_dim + epsilon);
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        // same rounding order as the reference: normalize in fp32, cast to half, then scale by weight in half
        out_row[i] = __hmul(weight[i], __float2half(__half2float(input_row[i]) / rms));
    }
}

// 2. The Host Wrapper (Called by C++)
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.device().is_cuda(), "Tensor x must be on a CUDA device");
    TORCH_CHECK(weight.device().is_cuda(), "Tensor weight must be on a CUDA device");
    TORCH_CHECK(weight.is_contiguous(), "Tensor weight must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf, "x and weight must be fp16");
    TORCH_CHECK(weight.numel() == x.size(-1), "weight must have hidden_dim elements");

    x = x.contiguous();  // rows must sit back-to-back for blockIdx.x * hidden_dim indexing
    auto c = torch::empty_like(x);

    int hidden = x.size(-1);
    int rows = x.numel() / hidden;
    int threads = 256;  // must be a power of two for the tree reduction

    // one block per row, on PyTorch's current stream (required for CUDA graph capture later)
    rms_norm_kernel<<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        eps,
        hidden
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}
