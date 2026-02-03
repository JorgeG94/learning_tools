#!/usr/bin/env python
"""Test script for GPU MRE - Minimal Reproducible Example.

Usage:
    # Single rank
    python test_mre.py

    # Multiple ranks
    mpirun -n 4 python test_mre.py

    # Force GPU offload
    OMP_TARGET_OFFLOAD=mandatory mpirun -n 4 python test_mre.py
"""

import numpy as np
from mpi4py import MPI

import gpu_mre


def test_saxpy():
    """Test SAXPY computation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize MRE
    ctx = gpu_mre.init(comm)

    print(f"[Rank {rank}] Initialized: {ctx}")

    # Create test arrays
    n = 1000000
    alpha = 2.0

    x = np.ones(n, dtype=np.float64) * (rank + 1)  # Rank-specific values
    y = np.ones(n, dtype=np.float64) * 10.0
    y_expected = alpha * x + y

    # Synchronize before test
    gpu_mre.barrier()

    # Run SAXPY
    gpu_mre.saxpy(alpha, x, y)

    # Synchronize after test
    gpu_mre.barrier()

    # Verify result
    max_error = np.max(np.abs(y - y_expected))
    all_close = np.allclose(y, y_expected)

    print(f"[Rank {rank}] SAXPY result: max_error={max_error:.2e}, correct={all_close}")

    # Get device info
    device_info = gpu_mre.get_device_info()
    print(f"[Rank {rank}] Device info: {device_info}")

    # Final barrier
    gpu_mre.barrier()

    # Cleanup
    gpu_mre.finalize()

    if rank == 0:
        print("\n=== Test Summary ===")
        print(f"Array size: {n}")
        print(f"Alpha: {alpha}")
        print(f"All ranks completed successfully")

    return all_close


def main():
    success = test_saxpy()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Gather all results to rank 0
    all_success = comm.gather(success, root=0)

    if rank == 0:
        if all(all_success):
            print("\n*** ALL TESTS PASSED ***")
        else:
            print("\n*** SOME TESTS FAILED ***")
            exit(1)


if __name__ == "__main__":
    main()
