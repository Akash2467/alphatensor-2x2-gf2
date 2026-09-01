"""Finite action catalog for the 2 x 2 TensorGame over GF(2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decomposition import Factor
from .tensor import GF2Tensor


VECTOR_SIZE = 4
NONZERO_VECTOR_COUNT = (1 << VECTOR_SIZE) - 1
ACTION_COUNT = NONZERO_VECTOR_COUNT**3


def vector_from_code(code: int) -> NDArray[np.uint8]:
    """Decode an integer from 1 through 15 into a length-four GF(2) vector."""
    if not 1 <= code <= NONZERO_VECTOR_COUNT:
        raise ValueError(f"code must be between 1 and {NONZERO_VECTOR_COUNT}")
    return np.array(
        [(code >> shift) & 1 for shift in reversed(range(VECTOR_SIZE))],
        dtype=np.uint8,
    )


def code_from_vector(vector: ArrayLike) -> int:
    """Encode a nonzero length-four GF(2) vector as an integer from 1 to 15."""
    values = np.asarray(vector)
    if values.shape != (VECTOR_SIZE,):
        raise ValueError(f"vector must have shape ({VECTOR_SIZE},)")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("a GF(2) vector may contain only 0 and 1")
    if not np.any(values):
        raise ValueError("vector must be nonzero")

    code = 0
    for value in values:
        code = (code << 1) | int(value)
    return code


class GF2ActionSpace:
    """All nonzero ``(u, v, w)`` actions for four-dimensional GF(2) vectors.

    Actions use a stable mixed-radix ordering. The corresponding rank-one
    tensors are precomputed once, making state expansion cheap during search.
    """

    size = ACTION_COUNT

    def __init__(self) -> None:
        self._vectors = np.stack(
            [vector_from_code(code) for code in range(1, NONZERO_VECTOR_COUNT + 1)]
        )
        self._terms = np.einsum(
            "ai,bj,ck->abcijk", self._vectors, self._vectors, self._vectors
        ).reshape(ACTION_COUNT, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE)
        self._terms = self._terms.astype(np.uint8, copy=False)

    @property
    def vectors(self) -> NDArray[np.uint8]:
        """Return copies of the 15 possible nonzero vectors."""
        return self._vectors.copy()

    @property
    def terms(self) -> GF2Tensor:
        """Return a copy of every rank-one action tensor."""
        return self._terms.copy()

    def action(self, index: int) -> Factor:
        """Decode an action index into copies of ``(u, v, w)``."""
        u_position, v_position, w_position = self._positions(index)
        return (
            self._vectors[u_position].copy(),
            self._vectors[v_position].copy(),
            self._vectors[w_position].copy(),
        )

    def term(self, index: int) -> GF2Tensor:
        """Return a copy of the rank-one tensor for an action index."""
        self._validate_index(index)
        return self._terms[index].copy()

    def index(self, action: Factor) -> int:
        """Encode an action triplet as its stable integer index."""
        try:
            u, v, w = action
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a triplet (u, v, w)") from error

        u_position = code_from_vector(u) - 1
        v_position = code_from_vector(v) - 1
        w_position = code_from_vector(w) - 1
        return (
            u_position * NONZERO_VECTOR_COUNT**2
            + v_position * NONZERO_VECTOR_COUNT
            + w_position
        )

    @staticmethod
    def _validate_index(index: int) -> None:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < ACTION_COUNT:
            raise IndexError(f"action index must be between 0 and {ACTION_COUNT - 1}")

    def _positions(self, index: int) -> tuple[int, int, int]:
        self._validate_index(index)
        u_position, remainder = divmod(int(index), NONZERO_VECTOR_COUNT**2)
        v_position, w_position = divmod(remainder, NONZERO_VECTOR_COUNT)
        return u_position, v_position, w_position
