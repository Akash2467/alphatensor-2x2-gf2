"""Rank-preserving basis symmetries for 4 x 4 x 4 tensors over GF(2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .actions import GF2ActionSpace
from .game import slice_rank_upper_bound
from .search import SearchTrajectory
from .tensor import GF2Tensor


GF2Matrix = NDArray[np.uint8]
BasisTransform = tuple[GF2Matrix, GF2Matrix, GF2Matrix]


def gf2_inverse(matrix: ArrayLike) -> GF2Matrix:
    """Return the inverse of a square binary matrix over GF(2)."""
    values = np.asarray(matrix, dtype=np.uint8)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if np.any(values > 1):
        raise ValueError("matrix entries must be binary")

    size = values.shape[0]
    augmented = np.concatenate((values.copy(), np.eye(size, dtype=np.uint8)), axis=1)
    for column in range(size):
        pivots = np.flatnonzero(augmented[column:, column])
        if pivots.size == 0:
            raise ValueError("matrix is not invertible over GF(2)")
        pivot = column + int(pivots[0])
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        for row in range(size):
            if row != column and augmented[row, column]:
                augmented[row] ^= augmented[column]
    return augmented[:, size:]


def random_invertible_matrix(
    rng: np.random.Generator, *, size: int = 4
) -> GF2Matrix:
    """Sample a uniformly proposed invertible binary matrix."""
    if size < 1:
        raise ValueError("size must be positive")
    while True:
        matrix = rng.integers(0, 2, size=(size, size), dtype=np.uint8)
        try:
            gf2_inverse(matrix)
        except ValueError:
            continue
        return matrix


def random_basis_transform(rng: np.random.Generator) -> BasisTransform:
    """Sample independent invertible transforms for all three tensor modes."""
    return tuple(random_invertible_matrix(rng) for _ in range(3))  # type: ignore[return-value]


def transform_tensor(tensor: ArrayLike, transform: BasisTransform) -> GF2Tensor:
    """Apply independent changes of basis to a 4 x 4 x 4 GF(2) tensor."""
    values = np.asarray(tensor, dtype=np.uint8)
    if values.shape != (4, 4, 4) or np.any(values > 1):
        raise ValueError("tensor must be a binary array with shape (4, 4, 4)")
    a, b, c = transform
    for matrix in transform:
        if np.asarray(matrix).shape != (4, 4):
            raise ValueError("each basis matrix must have shape (4, 4)")
        gf2_inverse(matrix)
    transformed = np.einsum("ai,bj,ck,ijk->abc", a, b, c, values)
    return np.remainder(transformed, 2).astype(np.uint8)


def transform_action_index(
    action_index: int,
    transform: BasisTransform,
    *,
    action_space: GF2ActionSpace | None = None,
) -> int:
    """Transform a rank-one action and return its catalog index."""
    catalog = action_space or GF2ActionSpace()
    action = catalog.action(action_index)
    transformed = tuple(
        np.remainder(matrix @ vector, 2).astype(np.uint8)
        for matrix, vector in zip(transform, action)
    )
    return catalog.index(transformed)  # type: ignore[arg-type]


def transform_trajectory(
    trajectory: SearchTrajectory,
    transform: BasisTransform,
    *,
    action_space: GF2ActionSpace | None = None,
) -> SearchTrajectory:
    """Map every action and the terminal residual through one symmetry."""
    catalog = action_space or GF2ActionSpace()
    residual = transform_tensor(trajectory.residual, transform)
    return SearchTrajectory(
        action_indices=tuple(
            transform_action_index(index, transform, action_space=catalog)
            for index in trajectory.action_indices
        ),
        residual=residual,
        score=trajectory.score,
        residual_nonzero=int(np.count_nonzero(residual)),
        residual_rank_upper_bound=slice_rank_upper_bound(residual),
    )
