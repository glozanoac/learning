cmake_minimum_required(VERSION 3.16)
project(yupana LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# --- library ---
add_library(yupana
    src/csr.c
    #src/spmv_kernels.c
)
add_library(yupana::yupana ALIAS yupana)

target_include_directories(yupana PUBLIC include)
target_link_libraries(yupana PUBLIC OpenMP::OpenMP_C)
target_compile_options(yupana PRIVATE -Wall -Wextra)

# --- hello world (links against the library) ---
add_executable(hello src/hello.c)
target_link_libraries(hello PRIVATE yupana::yupana)
