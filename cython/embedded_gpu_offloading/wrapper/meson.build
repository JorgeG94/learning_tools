py3 = import('python').find_installation()


py3.extension_module(
  'wrapper',
  sources: ['wrapper.pyx'],
  dependencies: openmp_deps,
  c_args: openmp_c_args,
  link_args: openmp_c_args,
  install: true,
)

