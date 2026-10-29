cmake_minimum_required(VERSION 3.18)
project(fastla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# 1) Pure C++ library (no Python): reusable by C++ tests/benchmarks.
add_library(fastla_core STATIC src/fastla_core/vector_ops.cpp)
target_include_directories(fastla_core PUBLIC src)
set_target_properties(fastla_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fastla_core PRIVATE -O3 -march=native)
endif()

# 2) Python extension module: fastla._core
pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE fastla_core)

# Installed into the fastla/ package when built by pip (scikit-build-core).
install(TARGETS _core DESTINATION fastla)
