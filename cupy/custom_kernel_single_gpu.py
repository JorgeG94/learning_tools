'''
This file loads a cuda file, compiles it and runs it from python
'''
import cupy as cp

# Read the full CUDA source from a file
with open("hello.cu", "r") as f:
    cuda_code = f.read()

# Compile and load the CUDA kernel
mod = cp.RawModule(code=cuda_code,
                   options=("--std=c++17",),
                   name_expressions=["hello_from_gpu"])

# Get the function
kernel = mod.get_function("hello_from_gpu")

# Launch the kernel
kernel((1,), (1,), (), stream=cp.cuda.Stream.null)
cp.cuda.Device().synchronize()
