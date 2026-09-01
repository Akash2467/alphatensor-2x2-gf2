"""Execute and describe discovered 2 x 2 GF(2) multiplication algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decomposition import Factor


def apply_decomposition(
    left: ArrayLike, right: ArrayLike, factors: Sequence[Factor]
) -> NDArray[np.uint8]:
    """Multiply two binary 2 x 2 matrices using rank-one tensor factors."""
    left_array = np.asarray(left, dtype=np.uint8)
    right_array = np.asarray(right, dtype=np.uint8)
    if left_array.shape != (2, 2) or right_array.shape != (2, 2):
        raise ValueError("left and right matrices must have shape (2, 2)")
    if np.any(left_array > 1) or np.any(right_array > 1):
        raise ValueError("matrix entries must be binary")
    left_flat = left_array.reshape(4)
    right_flat = right_array.reshape(4)
    output = np.zeros(4, dtype=np.uint8)
    for u, v, w in factors:
        u_array = np.asarray(u, dtype=np.uint8)
        v_array = np.asarray(v, dtype=np.uint8)
        w_array = np.asarray(w, dtype=np.uint8)
        product = int(np.dot(u_array, left_flat) % 2) * int(
            np.dot(v_array, right_flat) % 2
        )
        if product:
            output ^= w_array
    return output.reshape(2, 2)


def exhaustive_verify_2x2(factors: Sequence[Factor]) -> tuple[bool, int]:
    """Check all 16 x 16 binary input-matrix pairs."""
    checked = 0
    for left_code in range(16):
        left = np.array(
            [(left_code >> shift) & 1 for shift in range(3, -1, -1)],
            dtype=np.uint8,
        ).reshape(2, 2)
        for right_code in range(16):
            right = np.array(
                [(right_code >> shift) & 1 for shift in range(3, -1, -1)],
                dtype=np.uint8,
            ).reshape(2, 2)
            expected = np.remainder(left @ right, 2).astype(np.uint8)
            actual = apply_decomposition(left, right, factors)
            checked += 1
            if not np.array_equal(actual, expected):
                return False, checked
    return True, checked


def describe_equations(factors: Sequence[Factor]) -> list[str]:
    """Return human-readable GF(2) scalar-product equations."""
    left_names = ("a11", "a12", "a21", "a22")
    right_names = ("b11", "b12", "b21", "b22")
    equations: list[str] = []
    for number, (u, v, _) in enumerate(factors, start=1):
        left_form = " + ".join(
            name for name, bit in zip(left_names, u) if int(bit)
        )
        right_form = " + ".join(
            name for name, bit in zip(right_names, v) if int(bit)
        )
        equations.append(f"p{number} = ({left_form})({right_form})")
    output_names = ("c11", "c12", "c21", "c22")
    for coordinate, output_name in enumerate(output_names):
        products = [
            f"p{number}"
            for number, (_, _, w) in enumerate(factors, start=1)
            if int(np.asarray(w)[coordinate])
        ]
        equations.append(f"{output_name} = " + " + ".join(products))
    return equations
