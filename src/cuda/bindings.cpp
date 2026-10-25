#include <torch/extension.h>

// Bindings for the custom-kernel backend (loaded by src/ops.py:load_custom_ext).
// Every *.cu file in src/cuda is compiled into this extension.
//
// To swap in a kernel:
//   1. Write it in a new .cu file with a host wrapper (see vector_add_kernel.cu).
//   2. Forward-declare the wrapper below and m.def() it under one of these names.
//      The signature must match the PyTorch reference in src/ops.py:
//
//        rms_norm(x, weight, eps)  -> Tensor
//        linear(x, weight)         -> Tensor          // x @ weight.T
//        silu_mul(gate, up)        -> Tensor
//        rope(q, k, cos, sin)      -> (Tensor, Tensor)
//        attention(q, k, v)        -> Tensor
//
// End each host wrapper with C10_CUDA_KERNEL_LAUNCH_CHECK() (from <c10/cuda/CUDAException.h>)
// so a failed launch raises instead of silently returning garbage.
//
// Any name not defined here falls back to PyTorch, so you can do one at a time.
// Then run: python benchmarks/test_ops.py && python benchmarks/test_baseline.py

// torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("rms_norm", &rms_norm_cuda, "Fused RMSNorm");
}
