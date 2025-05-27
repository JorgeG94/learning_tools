'''
Now this is an explicit attempt on using cupys
pinned memory pool for transfers. 

This was equally as performant as the other 
one since the heuristics in cupy are pretty good. 

'''
import numpy as np
import cupy as cp
import time

# Matrix size
N = 20000
nbytes = N * N * 8  # float64 = 8 bytes

# Set up pinned memory pool (CuPy default one)
pinned_pool = cp.get_default_pinned_memory_pool()
memory_pool = cp.get_default_memory_pool()

A_host = np.ndarray((N, N), dtype=np.float64)
B_host = np.ndarray((N, N), dtype=np.float64)
C_gpu = cp.ndarray((N, N), dtype=np.float64)

print(memory_pool.used_bytes())              # 0
print(memory_pool.total_bytes())             # 0
print(pinned_pool.n_free_blocks())    # 0

transfer_start = time.time()
A_gpu = cp.array(A_host)
B_gpu = cp.array(B_host)
transfer_end = time.time()
transfer_time = transfer_end - transfer_start
print(f"Host to Device Transfer Time (Pinned): {transfer_time:.6f} s")
print(f" used bytes {memory_pool.used_bytes()}")
print(f" free blocks {pinned_pool.n_free_blocks()}")


print("Start actual compute")
start_time = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device().synchronize()
end_time = time.time()

elapsed = end_time - start_time
gflops = (2 * N**3) / (elapsed * 1e9)

# Output results
print(f"Matrix size: {N} x {N}")
print(f"Compute Time: {elapsed:.6f} s")
print(f"GFLOP/s:      {gflops:.2f}")


