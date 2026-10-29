"""Type stubs for the compiled module fastla._core."""
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

_ArrayLike = Union[npt.NDArray[np.floating], Sequence[float]]

def dot(x: _ArrayLike, y: _ArrayLike) -> float:
    """Return x^T y."""

def norm2(x: _ArrayLike) -> float:
    """Return ||x||_2."""

def axpy(alpha: float, x: _ArrayLike, y: npt.NDArray[np.float64]) -> None:
    """In place: y <- alpha*x + y (y must be a contiguous float64 array)."""
