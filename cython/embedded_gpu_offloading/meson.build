project('testo', 'c', 'cpp', 'cython')

openmp = dependency('openmp', required: false)
if openmp.found()

  #openmp_c_args = ['-O3', '-march=native', '-fopenmp', '-g']
  openmp_c_args = ['-O3', '-fast', '-mp=gpu', '-Minfo=mp,accel', '-g']
  # openmp_c_args = ['-O3', '-march=native','-funroll-loops', '-fvectorize', '-Rpass=loop-vectorize', '-Rpass=loop-unroll', '-g'],
  # openmp_c_args = ['-O3', '-march=native', openmp_flag, '-g'],

  openmp_deps =  [openmp]

else
  openmp_deps = testlib_dep 
endif

#subdir('library')
subdir('wrapper')

