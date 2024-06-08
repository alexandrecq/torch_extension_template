# Torch Extension Template
An minimal template for implementing a custom PyTorch extension in CUDA.

This example shows how to compute the forward and backward passes of a linear layer. It is not intended to exceed the performance of the built-in Linear module, but to serve as a starting point for more complex PyTorch modules that benefit from a fully-fused kernel.

Based on [this pytorch tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html).

# Compiling
To compile the extension, there are two methods:
* Compile and install the extension "ahead of time" by running:
```
python setup.py install
```
* Or compile "just in time" by importing the `_C` backend in python:
```
from .csrc import _C
```
