'''
This file loads a cuda file, compiles it and runs it from python
using however many devices it can see
'''
import cupy as cp


# Read the full CUDA source from a file
with open("hello.cu", "r") as f:
    cuda_code = f.read()

# Compile and load the CUDA kernel
mod = cp.RawModule(code=cuda_code,
                   options=("--std=c++17",),
                   name_expressions=["hello_from_gpu"])


kernel = mod.get_function("hello_from_gpu")
# Launch the kernel with 1 block of 1 thread
kernel((1,), (1,), (), stream=cp.cuda.Stream.null)
cp.cuda.Device().synchronize()

# Get how many devices we have
n_devices = cp.cuda.runtime.getDeviceCount()
print(f"Found {n_devices} devices")

for i in range(n_devices):
    with cp.cuda.Device(i):
        print(f"Launching kernel on device {i}")

        # Load the module AFTER switching device
        mod = cp.cuda.function.Module()
        mod.load_file("hello.cubin")
        kernel = mod.get_function("hello_from_gpu")

        kernel((1,), (1,), (), stream=cp.cuda.Stream.null)
        cp.cuda.Device(i).synchronize()
