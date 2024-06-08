#include <torch/extension.h>
#include <vector>

using std::vector, torch::Tensor;

// CUDA forward declarations
Tensor linear_cuda_forward(
    Tensor input,
    Tensor weights,
    Tensor bias);

vector<Tensor> linear_cuda_backward(
    Tensor grad_output,
    Tensor input,
    Tensor weights);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor linear_forward(
    Tensor input,
    Tensor weights,
    Tensor bias)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return linear_cuda_forward(input, weights, bias);
}

vector<Tensor> linear_backward(
    Tensor grad_output,
    Tensor input,
    Tensor weights)
{
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return linear_cuda_backward(grad_output, input, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "linear forward (CUDA)");
  m.def("backward", &linear_backward, "linear backward (CUDA)");
}
