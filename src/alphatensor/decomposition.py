"""Exact tensor-decomposition utilities for arithmetic over GF(2)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .tensor import GF2Tensor, rank_one_tensor


Factor: TypeAlias = tuple[ArrayLike, ArrayLike, ArrayLike]


def compose_decomposition(
    factors: Iterable[Factor], *, vector_size: int = 4
) -> GF2Tensor:
    """Sum rank-one factors and return their tensor over GF(2).

    Repeated entries cancel because addition in GF(2) is XOR.
    """
    if vector_size < 1:
        raise ValueError("vector_size must be a positive integer")

    result = np.zeros((vector_size, vector_size, vector_size), dtype=np.uint8)
    for action_number, (u, v, w) in enumerate(factors, start=1):
        term = rank_one_tensor(u, v, w)
        if term.shape != result.shape:
            raise ValueError(
                f"factor {action_number} has vector length {term.shape[0]}; "
                f"expected {vector_size}"
            )
        result ^= term

    return result


def decomposition_residual(target: ArrayLike, factors: Iterable[Factor]) -> GF2Tensor:
    """Return the part of ``target`` left after applying all factors."""
    target_array = np.asarray(target, dtype=np.uint8)
    if target_array.ndim != 3 or not (
        target_array.shape[0] == target_array.shape[1] == target_array.shape[2]
    ):
        raise ValueError("target must be a cubic three-dimensional tensor")
    if np.any(target_array > 1):
        raise ValueError("a GF(2) target may contain only 0 and 1")

    decomposition = compose_decomposition(
        factors, vector_size=target_array.shape[0]
    )
    return target_array ^ decomposition


def verify_decomposition(target: ArrayLike, factors: Iterable[Factor]) -> bool:
    """Return whether ``factors`` reconstruct ``target`` exactly over GF(2)."""
    return not np.any(decomposition_residual(target, factors))
