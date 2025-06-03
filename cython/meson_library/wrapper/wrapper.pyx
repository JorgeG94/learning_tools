# wrapper.pyx
cdef extern from "test.h":
    void test_function()

def call_c_function_from_python():
    test_function()

