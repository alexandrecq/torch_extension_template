#include <torch/extension.h>
#include <vector>

at::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) 
{
    torch::Tensor output = torch::addmm(bias, input, weights.t());
    return output;
}

std::vector<torch::Tensor> linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights) 
{
    torch::Tensor grad_weights = grad_output.t().mm(input);
    torch::Tensor grad_bias = grad_output.sum(0, true);
    return {grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "linear forward");
  m.def("backward", &linear_backward, "linear backward");
}
