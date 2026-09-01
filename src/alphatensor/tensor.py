"""Tensor construction utilities for 2 x 2 matrix multiplication over GF(2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


GF2Tensor = NDArray[np.uint8]


def matrix_multiplication_tensor(n: int = 2) -> GF2Tensor:
    """Return the tensor encoding multiplication of two n x n matrices.

    Matrix entries are flattened in row-major order. The resulting tensor has
    shape ``(n**2, n**2, n**2)`` and is intended for exact arithmetic over
    GF(2).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    size = n * n
    tensor = np.zeros((size, size, size), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                a_index = i * n + j
                b_index = j * n + k
                c_index = i * n + k
                tensor[a_index, b_index, c_index] = 1

    return tensor


def rank_one_tensor(u: ArrayLike, v: ArrayLike, w: ArrayLike) -> GF2Tensor:
    """Construct ``u outer v outer w`` using exact GF(2) coefficients."""
    vectors = [np.asarray(vector, dtype=np.uint8) for vector in (u, v, w)]
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("u, v, and w must all be one-dimensional vectors")
    if not (len(vectors[0]) == len(vectors[1]) == len(vectors[2])):
        raise ValueError("u, v, and w must have equal lengths")
    if any(np.any(vector > 1) for vector in vectors):
        raise ValueError("GF(2) vectors may contain only 0 and 1")

    u_array, v_array, w_array = vectors
    return np.einsum("i,j,k->ijk", u_array, v_array, w_array) % 2
