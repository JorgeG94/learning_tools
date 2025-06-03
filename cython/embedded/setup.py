from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np 

ext = Extension(
    name="wrapper",
    sources=["wrapper.pyx", "test.c"],  # test.c is compiled in!
    include_dirs=["np.get_include()","."],                 # directory with test.h
    extra_compile_args=["-O3"]
)

setup(
    ext_modules=cythonize([ext])
)

