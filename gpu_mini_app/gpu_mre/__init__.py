"""GPU MRE - Minimal Reproducible Example for MPICH/mpi4py GPU offloading debugging."""

from .core import (
    init,
    finalize,
    saxpy,
    get_device_info,
    barrier,
)

__all__ = [
    'init',
    'finalize',
    'saxpy',
    'get_device_info',
    'barrier',
]
