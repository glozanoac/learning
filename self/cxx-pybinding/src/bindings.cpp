#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "fastla_core/vector_ops.hpp"

namespace py = pybind11;

// Accept only contiguous float64 arrays; forcecast converts lists/ints once.
using Vec = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void check_1d_same(const py::buffer_info& a, const py::buffer_info& b) {
    if (a.ndim != 1 || b.ndim != 1) throw std::invalid_argument("expected 1-D arrays");
    if (a.shape[0] != b.shape[0]) throw std::invalid_argument("size mismatch");
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "fastla: C++ linear algebra kernels";

    m.def("dot", [](Vec x, Vec y) {
        auto bx = x.request(), by = y.request();
        check_1d_same(bx, by);
        const double* px = static_cast<const double*>(bx.ptr);
        const double* py_ = static_cast<const double*>(by.ptr);
        py::gil_scoped_release release;  // let other Python threads run
        return fastla::dot(px, py_, static_cast<std::size_t>(bx.shape[0]));
    }, py::arg("x"), py::arg("y"), "Return x^T y.");

    m.def("norm2", [](Vec x) {
        auto bx = x.request();
        if (bx.ndim != 1) throw std::invalid_argument("expected 1-D array");
        return fastla::norm2(static_cast<const double*>(bx.ptr),
                             static_cast<std::size_t>(bx.shape[0]));
    }, py::arg("x"), "Return ||x||_2.");

    // y is modified in place: no forcecast, so a copy can never silently happen.
    m.def("axpy", [](double alpha, Vec x,
                     py::array_t<double, py::array::c_style> y) {
        auto bx = x.request(), by = y.request(true);
        check_1d_same(bx, by);
        fastla::axpy(alpha, static_cast<const double*>(bx.ptr),
                     static_cast<double*>(by.ptr),
                     static_cast<std::size_t>(bx.shape[0]));
    }, py::arg("alpha"), py::arg("x"), py::arg("y").noconvert(),
       "In place: y <- alpha*x + y (y must be a contiguous float64 array).");
}
