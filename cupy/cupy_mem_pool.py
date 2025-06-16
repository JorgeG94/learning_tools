import cupy as cp
import numpy as np

# Set up a custom pinned memory pool
pinned_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)

# Transfer from NumPy to CuPy (uses pinned memory underneath)
np_array = np.ones((10000, 10000), dtype=np.float64)

# Transfer to device — uses pinned memory under the hood
cp_array = cp.asarray(np_array)

# Manually delete to release pinned memory
del cp_array

# Force CuPy to clean up (garbage collect)
import gc
gc.collect()

# Now check free blocks
print(f"Free pinned blocks: {pinned_pool.n_free_blocks()}")

