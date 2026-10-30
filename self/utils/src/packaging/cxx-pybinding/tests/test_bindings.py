"""Tests for the Python bindings: values, conversions, in-place semantics, errors."""
import numpy as np
import pytest

import fastla

rng = np.random.default_rng(0)


def test_dot_matches_numpy():
    x, y = rng.standard_normal(1000), rng.standard_normal(1000)
    assert fastla.dot(x, y) == pytest.approx(x @ y)


def test_norm2_matches_numpy():
    x = rng.standard_normal(1000)
    assert fastla.norm2(x) == pytest.approx(np.linalg.norm(x))


def test_accepts_lists_and_ints():
    assert fastla.dot([1, 2, 3], [4, 5, 6]) == 32.0


def test_axpy_is_in_place():
    x, y = rng.standard_normal(100), rng.standard_normal(100)
    expected = 2.0 * x + y
    y_id = id(y)
    fastla.axpy(2.0, x, y)
    assert id(y) == y_id
    np.testing.assert_allclose(y, expected)


def test_axpy_rejects_non_float64_y():
    # A float32 y would need a copy, so the update would be lost: must fail.
    with pytest.raises(TypeError):
        fastla.axpy(1.0, np.ones(3), np.ones(3, dtype=np.float32))


def test_size_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="size mismatch"):
        fastla.dot(np.ones(3), np.ones(4))


def test_2d_input_raises_valueerror():
    with pytest.raises(ValueError, match="1-D"):
        fastla.norm2(np.ones((2, 2)))
