import cython 
import numpy as np
cdef extern from "test.c" nogil:
    void test_function()


def call_c_function_from_python():
    with nogil:
        test_function()
