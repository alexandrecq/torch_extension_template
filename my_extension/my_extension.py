## based on:
## https://pytorch.org/tutorials/advanced/cpp_extension.html

import torch
from torch.autograd import Function

# import linear_cpp_precompiled  # requires running `setup.py`
from .csrc import _C, _CU  # loaded by csrc/__init__.py


def linear_forward(input, weights, bias):
    return input @ weights.T + bias


def linear_backward(grad_output, input, weights):
    d_weights = grad_output.T @ input
    d_bias = grad_output.sum(dim=0, keepdims=True)
    return d_weights, d_bias


class LinearPytorch(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights)
        output = linear_forward(input, weights, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        d_weights, d_bias = linear_backward(grad_output, input, weights)
        return None, d_weights, d_bias


class LinearCPP(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights)
        output = _C.forward(input, weights, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        d_weights, d_bias = _C.backward(grad_output, input, weights)
        return None, d_weights, d_bias


class LinearCuda(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights)
        output = _CU.forward(input, weights, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_weights, d_bias = _CU.backward(grad_output, input, weights)
        return None, d_weights, d_bias
