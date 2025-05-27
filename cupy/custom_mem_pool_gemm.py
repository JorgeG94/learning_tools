'''
Now this is the better one. 

Here I create a custom memory pool which I then proceed
to slice and reuse accordingly to my memory needs. 

This meant that all the memory is pinned at the start and 
there's no need to repin the memory. 

The pinning still takes some time, however it does
have a gigantic impact into the transfer time. 

This is the way to go, it seems
'''
import numpy as np
import cupy as cp
import time

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
pinned_pool = cp.get_default_pinned_memory_pool()
memory_pool = cp.get_default_memory_pool()
pinned_ptr = pinned_pool.malloc(total_nbytes)

# Create a single big NumPy array view on the pinned memory
full_host_buffer = np.frombuffer(pinned_ptr, dtype=dtype, count=total_elements)

# Slice the pinned buffer into A_host and B_host
A_host = full_host_buffer[:n_elements].reshape(shape)
B_host = full_host_buffer[n_elements:].reshape(shape)

# Fill pinned buffers with data
setup_start = time.time()
np.random.seed(0)
A_host[:] = np.random.rand(*shape)
B_host[:] = np.random.rand(*shape)
setup_end = time.time()
setup_time = setup_end - setup_start 
print(f"Setup Time : {setup_time:.6f} s")

# Check pinned pool usage
print("Before transfer:")
print(f" used bytes {memory_pool.used_bytes()}")
print(f" free pinned blocks {pinned_pool.n_free_blocks()}")

# Fast pinned transfer
transfer_start = time.time()
A_gpu = cp.asarray(A_host)
B_gpu = cp.asarray(B_host)
cp.cuda.Device().synchronize()
transfer_end = time.time()

transfer_time = transfer_end - transfer_start
print(f"Host to Device Transfer Time (Pinned): {transfer_time:.6f} s")
print(f" used bytes {memory_pool.used_bytes()}")
print(f" free pinned blocks {pinned_pool.n_free_blocks()}")

# Matrix multiply
print("Start actual compute")
start_time = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device().synchronize()
end_time = time.time()

# Compute GFLOPs
elapsed = end_time - start_time
gflops = (2 * N**3) / (elapsed * 1e9)

print(f"Matrix size: {N} x {N}")
print(f"Compute Time: {elapsed:.6f} s")
print(f"GFLOP/s:      {gflops:.2f}")

