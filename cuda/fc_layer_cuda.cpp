#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> fc_layer_cuda_forward(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor b);

std::vector<at::Tensor> fc_layer_cuda_backward(
  torch::Tensor A,
  torch::Tensor W,
  torch::Tensor b,
  torch::Tensor dZ,
  torch::Tensor dA,
  torch::Tensor dW,
  torch::Tensor db);

std::vector<torch::Tensor> fc_layer_forward(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor b) {
    return fc_layer_cuda_forward(Z, A, W, b);
}

std::vector<torch::Tensor> fc_layer_backward(
  torch::Tensor A,
  torch::Tensor W,
  torch::Tensor b,
  torch::Tensor dZ,
  torch::Tensor dA,
  torch::Tensor dW,
  torch::Tensor db) {
      return fc_layer_cuda_backward(A, W, b, dZ, dA, dW, db);
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fc_layer_forward, "FC_LAYER forward (CUDA)");
  m.def("backward", &fc_layer_backward, "FC_LAYER backward (CUDA)");
}
