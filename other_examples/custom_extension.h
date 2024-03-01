#include <torch/extension.h>

#include <iostream>


torch::Tensor my_custom_function_forward(torch::Tensor input);
torch::Tensor my_custom_function_backward(torch::Tensor grad_output);
