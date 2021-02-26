from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deform_cuda',
    ext_modules=[
        CUDAExtension('deform_cuda', [
            'deform_conv.cpp',
            'deform_conv_cuda.cu',
            'modulated_deform_conv_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
