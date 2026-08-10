#!/bin/bash 

nx=1024
ny=1024

./benchmark_serial ${nx} ${ny}
./benchmark_serial_dc ${nx} ${ny}
ACC_NUM_CORES=1 ./benchmark_multicore_dc ${nx} ${ny}
./benchmark_multicore_dc ${nx} ${ny}
./benchmark_gpu_dc ${nx} ${ny}
