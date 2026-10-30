"""fastla: C++ linear algebra kernels for optimization."""
from ._core import axpy, dot, norm2

__all__ = ["dot", "norm2", "axpy"]
__version__ = "0.1.0"
