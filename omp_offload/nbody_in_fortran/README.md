This is a simple nbody simulator that is accelerated with OpenMP. 

A similar implementation using CUDA takes 29 ms per iteration, this one 
on a V100 takes around 40 ms for 70,000 particles. 

Using the exact same code and 40 OpenMP threads each timestep takes 1s/iteration. This
represents thus a 26.8x speedup using one GPU.

Builkd using `nvfortran -gpu=fastmath -O4 -mp=gpu -gpu=mem:separate -gpu=maxregcount:80 -Minfo=all nbody_gpu.f90`
