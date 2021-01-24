from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fc_layer_cuda',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('fc_layer_cuda', [
            'fc_layer_cuda.cpp',
            'fc_layer_cuda_kernel.cu'])
            ],
    cmdclass={'build_ext': BuildExtension})