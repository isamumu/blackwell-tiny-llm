#include <torch/extension.h>

// Forward declaration of our CUDA wrapper function
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);

// Create the module entry point
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "CUDA Vector Addition (Separate Files Test)");
}
