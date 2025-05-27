'''
This example uses Caliper from LLNL (github.com/LLNL/Caliper)
to measure performance of the code in a more granular fashion. 

Simply a demonstration of it being used in a python file, could
be useful but we need a conda recipe so that we can load
it as a dependency.
'''
import numpy as np
import cupy as cp
import time

from pycaliper.instrumentation import (
    set_global_byname,
    begin_region,
    end_region,
)
from pycaliper.loop import Loop


# Matrix size
N = 20000
shape = (N, N)
dtype = np.float64
itemsize = np.dtype(dtype).itemsize
n_elements = np.prod(shape)

# Total size for two matrices A and B
total_elements = 2 * n_elements
total_nbytes = total_elements * itemsize

# Allocate a single pinned buffer
begin_region('allocation')
pinned_pool = cp.get_default_pinned_memory_pool()
memory_pool = cp.get_default_memory_pool()
pinned_ptr = pinned_pool.malloc(total_nbytes)
end_region('allocation')

# Create a single big NumPy array view on the pinned memory
full_host_buffer = np.frombuffer(pinned_ptr, dtype=dtype, count=total_elements)

# Slice the pinned buffer into A_host and B_host
A_host = full_host_buffer[:n_elements].reshape(shape)
B_host = full_host_buffer[n_elements:].reshape(shape)

# Fill pinned buffers with data
begin_region('random filling')
setup_start = time.time()
np.random.seed(0)
A_host[:] = np.random.rand(*shape)
B_host[:] = np.random.rand(*shape)
setup_end = time.time()
setup_time = setup_end - setup_start 
print(f"Setup Time : {setup_time:.6f} s")
end_region('random filling')

# Check pinned pool usage
print("Before transfer:")
print(f" used bytes {memory_pool.used_bytes()}")
print(f" free pinned blocks {pinned_pool.n_free_blocks()}")

# Fast pinned transfer
begin_region('memcpy')
transfer_start = time.time()
A_gpu = cp.asarray(A_host)
B_gpu = cp.asarray(B_host)
cp.cuda.Device().synchronize()
transfer_end = time.time()
end_region('memcpy')

transfer_time = transfer_end - transfer_start
print(f"Host to Device Transfer Time (Pinned): {transfer_time:.6f} s")
print(f" used bytes {memory_pool.used_bytes()}")
print(f" free pinned blocks {pinned_pool.n_free_blocks()}")

# Matrix multiply
print("Start actual compute")
begin_region('dgemm')
start_time = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device().synchronize()
end_time = time.time()
end_region('dgemm')

# Compute GFLOPs
elapsed = end_time - start_time
gflops = (2 * N**3) / (elapsed * 1e9)

print(f"Matrix size: {N} x {N}")
print(f"Compute Time: {elapsed:.6f} s")
print(f"GFLOP/s:      {gflops:.2f}")

