#include "fastla/vector_ops.hpp"
#include <cmath>

namespace fastla {

double dot(const double* x, const double* y, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

double norm2(const double* x, std::size_t n) {
    return std::sqrt(dot(x, x, n));
}

void axpy(double alpha, const double* x, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

}  // namespace fastla
