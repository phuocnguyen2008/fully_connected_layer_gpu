#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> fc_layer_forward(
    /*
    Computes the forward pass for fully-connected layer.

    The input has shape (N, D) and contains a minibatch of N examples, where each example x[i] has dimension D.

    Inputs:
    - input: x(N, D)
    - weights: W(D, M)
    - bias: b(M, )

    Return:
    - output: (N, M)
    - cache: (input, weights, bias)
    */
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    auto output = torch::addmm(bias, input, weights);
    
    return {output, input, weights, bias};
}

std::vector<at::Tensor> fc_layer_backward(
  /*
  Computes the backward pass for fully-connected layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: (input, weights, bias)

  Return:
  - dx: Gradient w.r.t input x, of shape (N, D)
  - dW: Gradient w.r.t weights W, of shape (D, M)
  - db: Gradient w.r.t bias b, of shape (M,)
  */

  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor dout) {
  auto WT = torch::transpose(weights, 0, 1);
  auto dx = torch::addmm(input, dout, WT, 0, 1);
  auto xT = torch::transpose(input, 0, 1);
  auto dW = torch::addmm(weights, xT, dout, 0, 1);
  auto db = torch::sum(dout, 0);

  return {dx, dW, db};

}

PYBIND11_MODULE(fc_layer, m) {
  m.def("forward", &fc_layer_forward, "FC_Layer forward");
  m.def("backward", &fc_layer_backward, "FC_Layer backward");
}