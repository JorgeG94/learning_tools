from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="wrapper",
        sources=["wrapper.pyx"],
        include_dirs=["."],         # directory with test.h
        library_dirs=["."],         # directory with libtest.a
        libraries=["test_lib"],         # links to libtest.a
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(ext_modules, language_level=3),
)

