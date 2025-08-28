# cuda aware mpi tests 

## Dependencies 

You need:

- a compiler 
- an mpi implementaiton (gpu enabled if you want the aware program to not segfault)

## How to build 

Simply: `make`

```
Available targets:
  all          - Build both versions (default)
  aware        - Build CUDA-aware version only
  unaware      - Build non-CUDA-aware version only
  clean        - Remove compiled binaries
  run-aware    - Build and run CUDA-aware version
  run-unaware  - Build and run non-CUDA-aware version
  run-both     - Build and run both versions
  debug        - Build with debug flags
  profile-*    - Profile with nsys (requires NVIDIA Nsight Systems)
  check-mpi    - Show MPI and CUDA configuration
  help         - Show this help message
```
