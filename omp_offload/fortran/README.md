# omp_offloading

## Dependencies: 

NVFORTRAN 

How to build:

```
fpm install --prefix . --profile release --flag "-O3 -fast -mp=gpu -Minfo=accel" --verbose
```

How to run from fpm:

```
fpm run --profile release --flag "-O3 -fast -mp=gpu -Minfo=accel" --verbose
```

```
fpm install --prefix . --flag "-O3 -mp=gpu -I/apps/nvidia-hpc-sdk/25.5/compilers/include/" -verbose --link-flag "-cudalib=nvtx"
```

module load cuda/12.8.0
module load nvidia-hpc-skd/25.5
FOr marshall and ed : nvfortran a.f90 -cudalib=nvtx
