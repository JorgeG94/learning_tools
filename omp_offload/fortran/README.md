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
