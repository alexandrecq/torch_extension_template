from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, load

setup(
    name='custom_extension',
    ext_modules=[
        CUDAExtension('custom_extension', [
            'custom_extension.cpp',
            'custom_extension.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

## setuptools version
# Extension(
#    name='linear_cpp',
#    sources=['linear.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++'
# )
