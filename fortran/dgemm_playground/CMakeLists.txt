cmake_minimum_required(VERSION 3.15)
project(DGEMM_BENCHMARK LANGUAGES Fortran C)

# Enable OpenMP
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# === Build the library ===
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source/lib)

add_library(dgemm_lib STATIC
    ${LIB_DIR}/lib_types.f90
    ${LIB_DIR}/lib_gemm.f90
)

# make sure the libraries see the OpenMP dependency
target_link_libraries(dgemm_lib PRIVATE OpenMP::OpenMP_Fortran BLAS::BLAS)


# Optional: put module files in clean directory

# === Build the executable ===
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)

add_executable(dgemm_benchmark ${SRC_DIR}/dgemm_benchmark.f90)
set(targets_gemm dgemm_lib dgemm_benchmark)

# loop over targets to make sure they all see the necessary things
set(compile_flags -O3)

foreach(target ${targets_gemm})
  set_target_properties(${target} PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules
    )
  if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND compile_flags -march=native -g) 
  elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "NVHPC")
    if(ENABLE_OMP_OFFLOAD)
    list(APPEND compile_flags -mp=gpu -Minfo=mp,accel -g -traceback)
    #list(APPEND compile_flags -mp -mp=gpu -gpu=cc70,lineinfo -Minfo=mp,accel -g -traceback)
    elseif(ENABLE_DC)
    list(APPEND compile_flags -g -stdpar=gpu,multicore)
    endif()

   endif()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: 
                                             ${compile_flags} >)
endforeach()

# for some reason we need to add the link options to pick up these variables, otherwise we don't link and we sad!
  if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
  target_link_options(
    dgemm_benchmark
    PRIVATE
    -g
    -fopenmp)
  elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "NVHPC")
if(ENABLE_OMP_OFFLOAD)
  target_link_options(
    dgemm_benchmark
    PRIVATE
    ${compile_flags})
  elseif(ENABLE_DC)
  target_link_options(
    dgemm_benchmark
    PRIVATE
    ${compile_flags}
    )
  endif()

  endif()

# Link library and OpenMP into executable
target_link_libraries(dgemm_benchmark PRIVATE dgemm_lib OpenMP::OpenMP_Fortran)

