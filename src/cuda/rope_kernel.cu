#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 1. The Device Code (Runs on GPU threads)
// One thread per output element, covering q and k in a single launch: indices [0, q_elements) are q,
// [q_elements, q_elements + k_elements) are k. q and k have different head counts (GQA: 32 vs 8),
// but share batch, sequence length T and head_dim D, so the same indexing works for both.
//
// Pairing follows HF's rotate_half: element d pairs with d + D/2 (not the adjacent element).
//   d <  D/2:  out[d] = x[d] * cos[d] - x[d + D/2] * sin[d]
//   d >= D/2:  out[d] = x[d] * cos[d] + x[d - D/2] * sin[d]
__global__ void rope_kernel(half* q_out, half* k_out, const half* q, const half* k,
                            const half* cos, const half* sin,
                            const int q_elements, const int k_elements, const int seq_len, const int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= q_elements + k_elements) return;

    // pick the tensor this thread works on
    const bool is_q = idx < q_elements;
    const half* x = is_q ? q : k;
    half* out = is_q ? q_out : k_out;
    if (!is_q) idx -= q_elements;

    // [B, H, T, D] contiguous: d = position within the head, t = token position
    int d = idx % head_dim;
    int t = (idx / head_dim) % seq_len;
    int half_dim = head_dim / 2;

    float cos_val = __half2float(cos[t * head_dim + d]);
    float sin_val = __half2float(sin[t * head_dim + d]);
    float x_val = __half2float(x[idx]);
    float rotated = d < half_dim ? -__half2float(x[idx + half_dim]) : __half2float(x[idx - half_dim]);

    out[idx] = __float2half(x_val * cos_val + rotated * sin_val);
}

// 2. The Host Wrapper (Called by C++)
std::tuple<torch::Tensor, torch::Tensor> rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin) {

    TORCH_CHECK(q.device().is_cuda(), "Tensor q must be on a CUDA device");
    TORCH_CHECK(k.device().is_cuda(), "Tensor k must be on a CUDA device");
    TORCH_CHECK(cos.device().is_cuda(), "Tensor cos must be on a CUDA device");
    TORCH_CHECK(sin.device().is_cuda(), "Tensor sin must be on a CUDA device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf, "q and k must be fp16");
    TORCH_CHECK(cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf, "cos and sin must be fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q and k must be [B, heads, T, head_dim]");
    // GQA: head counts (dim 1) may differ; everything else must match
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3),
                "q and k must share batch, sequence length and head_dim");

    const int seq_len = q.size(2);
    const int head_dim = q.size(3);
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(cos.numel() == seq_len * head_dim && sin.numel() == seq_len * head_dim,
                "cos and sin must be [T, head_dim]");

    // in the model q/k are transposed views; the kernel's indexing needs [B, H, T, D] laid out contiguously
    q = q.contiguous();
    k = k.contiguous();
    cos = cos.contiguous();
    sin = sin.contiguous();

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    const int q_elements = q.numel();
    const int k_elements = k.numel();
    int threads = 256;
    int blocks = (q_elements + k_elements + threads - 1) / threads;

    rope_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        q_elements,
        k_elements,
        seq_len,
        head_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {q_out, k_out};
}
