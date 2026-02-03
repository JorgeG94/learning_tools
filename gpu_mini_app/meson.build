project('gpu_mre', 'c', 'cython',
    version: '0.1.0',
    default_options: [
        'c_std=c11',
        'warning_level=2',
    ]
)

# Python module discovery
py = import('python').find_installation(pure: false)

# Get options
gpu_offload = get_option('gpu_offload')
gpu_arch = get_option('gpu_arch')

# Compiler detection
cc = meson.get_compiler('c')
compiler_id = cc.get_id()

# Base OpenMP dependency
omp_dep = dependency('openmp', required: true)

# Build compiler-specific flags for GPU offloading
c_args = []
link_args = []

if gpu_offload
    message('GPU offloading enabled with arch: ' + gpu_arch)

    if compiler_id == 'nvidia_hpc'
        # NVIDIA HPC SDK (nvc)
        c_args += ['-mp=gpu', '-gpu=' + gpu_arch]
        link_args += ['-mp=gpu', '-gpu=' + gpu_arch]
    elif compiler_id == 'clang' or compiler_id == 'clang-cl'
        # Clang with offloading
        c_args += ['-fopenmp-targets=nvptx64-nvidia-cuda', '--offload-arch=sm_70']
        link_args += ['-fopenmp-targets=nvptx64-nvidia-cuda', '--offload-arch=sm_70']
    elif compiler_id == 'gcc'
        # GCC with offloading (requires offloading-enabled build)
        c_args += ['-foffload=nvptx-none', '-foffload-options=-misa=sm_70']
        link_args += ['-foffload=nvptx-none']
    else
        warning('Unknown compiler for GPU offloading: ' + compiler_id)
    endif
else
    message('CPU-only mode (no GPU offloading)')
    c_args += ['-DCPU_ONLY_MODE']
endif

# Store in configuration for subdir
gpu_c_args = c_args
gpu_link_args = link_args

subdir('gpu_mre')
