cmake_minimum_required(VERSION 3.18)
project(fastla VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FASTLA_BUILD_PYTHON "Build the Python extension module" ON)
option(FASTLA_BUILD_TESTS  "Build the C++ unit tests (Catch2)"  OFF)
option(FASTLA_NATIVE       "Optimize for this machine's CPU (non-portable)" OFF)

# ---------------------------------------------------------------- C++ core
add_library(fastla_core STATIC cpp/src/vector_ops.cpp)
add_library(fastla::core ALIAS fastla_core)
target_include_directories(fastla_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)
set_target_properties(fastla_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fastla_core PRIVATE -O3 -Wall -Wextra)
  if(FASTLA_NATIVE)
    target_compile_options(fastla_core PRIVATE -march=native)
  endif()
endif()

# ---------------------------------------------------------- Python module
if(FASTLA_BUILD_PYTHON)
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/module.cpp)
  target_link_libraries(_core PRIVATE fastla::core)
  install(TARGETS _core DESTINATION fastla)   # lands inside the fastla package
endif()

# -------------------------------------------------------------- C++ tests
if(FASTLA_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1)
  FetchContent_MakeAvailable(Catch2)

  add_executable(fastla_tests cpp/tests/test_vector_ops.cpp)
  target_link_libraries(fastla_tests PRIVATE fastla::core Catch2::Catch2WithMain)

  include(CTest)
  include(Catch)
  catch_discover_tests(fastla_tests)
endif()
