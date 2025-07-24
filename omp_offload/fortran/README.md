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
