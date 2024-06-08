import os

from torch.utils.cpp_extension import load


current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_file_path = os.path.join(current_dir, 'linear.cpp')
cuda_sources = [os.path.join(current_dir, file_name)
                for file_name in ('linear_cuda.cpp', 'linear_cuda_kernel.cu')]

_C = load(name='linear_cpp', sources=[cpp_file_path])
_CU = load(name='linear_cuda', sources=cuda_sources)
