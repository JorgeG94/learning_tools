cmake_minimum_required(VERSION 3.22)
project(test_c_library C)

# Option to choose STATIC or SHARED library
option(BUILD_SHARED_LIBS "Build shared instead of static" OFF)

# Create the library
add_library(test_lib test.c)

# Optionally, export headers (useful for install rules)
target_include_directories(test_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS test_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin  # Optional, for Windows
)
install(FILES test.h DESTINATION include)
