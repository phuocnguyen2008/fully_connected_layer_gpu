from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fc_layer',
      version='0.0.1',
      ext_modules=[cpp_extension.CppExtension('fc_layer', ['fc_layer.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})