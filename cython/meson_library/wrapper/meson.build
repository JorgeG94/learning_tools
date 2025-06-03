py3 = import('python').find_installation()

inc_dir = include_directories('../library')

py3.extension_module(
  'wrapper',
  sources: ['wrapper.pyx'],
  include_directories: inc_dir,
  dependencies: [testlib_dep],
  install: true,
)

