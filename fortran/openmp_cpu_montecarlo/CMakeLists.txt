cmake_minimum_required(VERSION 3.10)
project(MonteCarloFortran LANGUAGES Fortran)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# First executable
add_executable(montecarlo_v0 montecarlo_v0.f90)
target_link_libraries(montecarlo_v0 PRIVATE OpenMP::OpenMP_Fortran)

# Second executable
add_executable(montecarlo_v1 montecarlo_v1.f90)
target_link_libraries(montecarlo_v1 PRIVATE OpenMP::OpenMP_Fortran)

