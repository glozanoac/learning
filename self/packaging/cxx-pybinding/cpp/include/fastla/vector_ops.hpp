#pragma once
#include <cstddef>

// Pure C++ kernels: no Python dependency, reusable from any C++ code.
namespace fastla {

// Returns x^T y.
double dot(const double* x, const double* y, std::size_t n);

// Returns ||x||_2.
double norm2(const double* x, std::size_t n);

// In place: y <- alpha * x + y.
void axpy(double alpha, const double* x, double* y, std::size_t n);

}  // namespace fastla
