#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename T>
inline T div_round_up(T val, T divisor) {
    return (val - 1) / divisor + 1;
}

template <typename scalar_t>
__global__ void linear_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    size_t batch_size,
    size_t n_in,
    size_t n_out) 
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size || col >= n_out) return;
    // output = input @ weights.T + bias
    scalar_t temp = 0;
    for (int k_ = 0; k_ < n_in; k_++) {
        temp += input[row * n_in + k_] * weights[col * n_in + k_];
    }
    output[row * n_out + col] = temp + bias[col];
}

torch::Tensor linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias)
{
  const int64_t batch_size = input.size(0);
  const int64_t n_in = input.size(1);
  const int64_t n_out = weights.size(0);

  auto options = torch::TensorOptions().dtype(input.dtype()).device(torch::kCUDA, 0);
  auto output = torch::zeros({batch_size, n_out}, options);

  const int64_t threads = 32;
  const dim3 threadsPerBlock(threads, threads);
  const dim3 blocks(div_round_up(batch_size, threads), div_round_up(n_out, threads));

  AT_DISPATCH_FLOATING_TYPES(input.type(), "linear_forward_cuda", ([&] {
    linear_cuda_forward_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
        input.data<scalar_t>(),
        weights.data<scalar_t>(),
        bias.data<scalar_t>(),
        output.data<scalar_t>(),
        batch_size,
        n_in,
        n_out);
  }));

  return output;
}

template <typename scalar_t>
__global__ void linear_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_bias,
    size_t batch_size,
    size_t n_in,
    size_t n_out) 
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out || col >= n_in) return;
    // grad_weights = grad_output.T @ input
    // grad_bias = grad_output.sum(0, true);
    scalar_t temp = 0;
    scalar_t bias_temp = 0;
    for (int k_ = 0; k_ < batch_size; k_++) {
        temp += grad_output[k_ * n_out + row] * input[k_ * n_in + col];
        if (col == 0) bias_temp += grad_output[row * batch_size + k_];
    }
    grad_weights[row * n_in + col] = temp;
    if (col == 0) grad_bias[row] = bias_temp;
}

std::vector<torch::Tensor> linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights)
{
    const int64_t batch_size = input.size(0);
    const int64_t n_in = input.size(1);
    const int64_t n_out = weights.size(0);

    auto grad_weights = torch::zeros_like(weights);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(torch::kCUDA, 0);
    auto grad_bias = torch::zeros(n_out, options);

    const int64_t threads = 32;
    const dim3 threadsPerBlock(threads, threads);
    const dim3 blocks(div_round_up(n_out, threads), div_round_up(n_in, threads));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "linear_backward_cuda", ([&] {
    linear_cuda_backward_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
        grad_output.data<scalar_t>(),
        input.data<scalar_t>(),
        weights.data<scalar_t>(),
        grad_weights.data<scalar_t>(),
        grad_bias.data<scalar_t>(),
        batch_size,
        n_in,
        n_out);
    }));

    return {grad_weights, grad_bias};
}
