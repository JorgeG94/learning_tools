'''
Ideally here I wanted to show how a non pinned memory transfer
in cupy works. 

However, I ended up showing that if I use cp.array(A_host) cupy 
knows this is an array and magically pins the memory 

Cool but could incur into spending waaaaaaay too much time in the memory
pinning 

I am aware that that is not the FLOP for DGEMM, but I am doing a m=n=k dgemm so 
I don't really care
'''
import numpy as np
import cupy as cp
import time

# Matrix size (adjust as needed for testing)
N = 20000

# Generate data on host (NumPy)
A_host = np.random.rand(N, N)
B_host = np.random.rand(N, N)

# Start timing (host to device + computation)

# Transfer to device (slow if not pinned)
transfer_start = time.time()
A_gpu = cp.array(A_host)
B_gpu = cp.array(B_host)
transfer_end = time.time()

transfer_time = transfer_end - transfer_start 
print(f"Time taken: {transfer_time:.6f} s")


print("Start actual compute")
# Perform matrix multiplication (DGEMM equivalent)
start_time = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
# Ensure all GPU operations are complete
cp.cuda.Device().synchronize()

end_time = time.time()
elapsed = end_time - start_time

# Compute FLOPs: 2*N^3 for dense matmul
gflops = (2 * N**3) / (elapsed * 1e9)

# Output results
print(f"Matrix size: {N} x {N}")
print(f"Time taken: {elapsed:.6f} s")
print(f"GFLOP/s:    {gflops:.2f}")

