testlib_sources = ['test.c']
testlib = static_library(
  'test_lib',
  testlib_sources,
  include_directories: include_directories('.')
)

testlib_dep = declare_dependency(
  include_directories: include_directories('.'),
  link_with: testlib  # <-- THIS FIXES THE ERROR
)
