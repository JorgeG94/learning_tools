# How to use 

This sample is mostly a small example for how to include gpu offloading flags into your CMake project. 

For gfortran on Gadi, the capability is built in. So we simply need to compile with `-fopenmp`. 

For nvfortran we need to specify the architecture and the flags are non standard. So, on Gadi, if you are using 
the `nvidia-hpc-sdk/25.1` you can simply do:

```
mkdir build
cd build
cmake -DENABLE_DC=ON ../
make -j
./dgemm 100 100 100 10 10
```

To execute a dgemm of a 100x100 matrix, 10 times, 10 times. 
