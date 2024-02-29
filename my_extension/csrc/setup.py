from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, load

setup(name='linear_cpp_precompiled',
      ext_modules=[CppExtension('linear_cpp', ['linear.cpp'])],
      cmdclass={'build_ext': BuildExtension}
)
