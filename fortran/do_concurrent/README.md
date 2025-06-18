# Sanity checks and benchmarks 

```
source load_gadi_env.sh
make
make check
```

The sanity check files contain the following

 - serial dgemm versus openmp offloaded dgemm 
 - serial dgemm versus do concurrent offloaded dgemm 

The benchmarks are simply one using OpenMP timing functions and the other using `SYSTEM_CLOCK`

Usage of benchmark script: `./dc_gpu m n k reps reps_of_reps`

Where m,n, and k are the sizes of the matrix. Reps counts the amount of times the dgemm is executed; and reps of reps executes that dgemm N more times. 

For example:

`./dc_gpu 10 10 10 10 10` will execute a dgemm of three mtrices that are 10x10, 10 times, 10 times.
