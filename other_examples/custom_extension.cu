#include <torch/extension.h>

torch::Tensor my_custom_function_forward(torch::Tensor input) {
    return input * 2;  // Example operation
}

torch::Tensor my_custom_function_backward(torch::Tensor grad_output) {
    auto mask = grad_output.clone();
    mask.masked_fill_(grad_output < 0, 0);  // Example backward operation
    return mask;
}
