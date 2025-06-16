import cupy as cp
pool = cp.cuda.PinnedMemoryPool()
gpu_pool = cp.cuda.MemoryPool()
cp.cuda.set_allocator(pool.malloc)
cp.cuda.set_allocator(gpu_pool.malloc)
