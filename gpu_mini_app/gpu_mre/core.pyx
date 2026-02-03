# distutils: language = c
# cython: language_level = 3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Import mpi4py C interface
from mpi4py import MPI
from mpi4py cimport MPI as MPI_
from mpi4py.libmpi cimport MPI_Comm, MPI_SUCCESS

np.import_array()

# External declarations from kernels.h
cdef extern from "kernels.h":
    ctypedef struct MREContext:
        int rank
        int size
        int num_devices
        int selected_device
        MPI_Comm comm

    int mre_init(MPI_Comm comm, MREContext* ctx)
    int mre_finalize(MREContext* ctx)
    int mre_saxpy(int n, double alpha, double* x, double* y)
    int mre_get_num_devices()
    int mre_get_selected_device()
    int mre_barrier(MREContext* ctx)

# Global context
cdef MREContext* g_ctx = NULL

def init(comm=None):
    """Initialize the MRE with an MPI communicator.

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator. Defaults to MPI.COMM_WORLD.

    Returns
    -------
    dict
        Context info with rank, size, num_devices, selected_device.
    """
    global g_ctx

    if comm is None:
        comm = MPI.COMM_WORLD

    cdef MPI_.Comm py_comm = <MPI_.Comm>comm
    cdef MPI_Comm c_comm = py_comm.ob_mpi

    if g_ctx != NULL:
        free(g_ctx)

    g_ctx = <MREContext*>malloc(sizeof(MREContext))
    if g_ctx == NULL:
        raise MemoryError("Failed to allocate MRE context")

    cdef int err = mre_init(c_comm, g_ctx)
    if err != MPI_SUCCESS:
        free(g_ctx)
        g_ctx = NULL
        raise RuntimeError(f"MRE init failed with error {err}")

    return {
        'rank': g_ctx.rank,
        'size': g_ctx.size,
        'num_devices': g_ctx.num_devices,
        'selected_device': g_ctx.selected_device,
    }

def finalize():
    """Finalize and cleanup the MRE context."""
    global g_ctx

    if g_ctx != NULL:
        mre_finalize(g_ctx)
        free(g_ctx)
        g_ctx = NULL

def saxpy(double alpha, double[::1] x not None, double[::1] y not None):
    """Compute y = alpha * x + y using GPU offloading.

    Parameters
    ----------
    alpha : float
        Scalar multiplier.
    x : ndarray
        Input array (contiguous, float64).
    y : ndarray
        Input/output array (contiguous, float64).
    """
    cdef int n = x.shape[0]

    if y.shape[0] != n:
        raise ValueError(f"Array size mismatch: x has {n} elements, y has {y.shape[0]}")

    cdef int err = mre_saxpy(n, alpha, &x[0], &y[0])
    if err != 0:
        raise RuntimeError(f"SAXPY failed with error {err}")

def get_device_info():
    """Get current device information.

    Returns
    -------
    dict
        Dictionary with num_devices and selected_device.
    """
    return {
        'num_devices': mre_get_num_devices(),
        'selected_device': mre_get_selected_device(),
    }

def barrier():
    """Execute MPI barrier on the initialized communicator."""
    global g_ctx

    if g_ctx == NULL:
        raise RuntimeError("MRE not initialized. Call init() first.")

    cdef int err = mre_barrier(g_ctx)
    if err != MPI_SUCCESS:
        raise RuntimeError(f"MPI barrier failed with error {err}")
