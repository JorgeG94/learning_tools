# MPI dependency
mpi_dep = dependency('mpi', language: 'c', required: true)

# NumPy include path
numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'],
                        check: true).stdout().strip()

# mpi4py include path
mpi4py_inc = run_command(py, ['-c', 'import mpi4py; print(mpi4py.get_include())'],
                         check: true).stdout().strip()

# Include directories
inc_dirs = include_directories(numpy_inc, mpi4py_inc)

# Build the C kernels as a static library
kernels_lib = static_library('kernels',
    'kernels.c',
    c_args: gpu_c_args,
    link_args: gpu_link_args,
    dependencies: [omp_dep, mpi_dep],
    include_directories: inc_dirs,
)

# Build the Cython extension module
py.extension_module('core',
    'core.pyx',
    c_args: gpu_c_args,
    link_args: gpu_link_args,
    link_with: kernels_lib,
    dependencies: [omp_dep, mpi_dep],
    include_directories: inc_dirs,
    install: true,
    subdir: 'gpu_mre',
)

# Install Python files
py.install_sources(
    '__init__.py',
    subdir: 'gpu_mre',
)
