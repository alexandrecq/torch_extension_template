from setuptools import setup

from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(name='linear_cpp_precompiled',
      ext_modules=[CppExtension('linear_cpp', ['linear.cpp'])],
      cmdclass={'build_ext': BuildExtension})
