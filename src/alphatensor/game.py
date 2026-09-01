"""Single-player TensorGame environment for the 2 x 2 GF(2) prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .decomposition import Factor
from .tensor import GF2Tensor, matrix_multiplication_tensor, rank_one_tensor


def gf2_matrix_rank(matrix: ArrayLike) -> int:
    """Compute an exact matrix rank using Gaussian elimination over GF(2)."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("a GF(2) matrix may contain only 0 and 1")

    reduced = values.astype(np.uint8, copy=True)
    rows, columns = reduced.shape
    rank = 0

    for column in range(columns):
        pivot_candidates = np.flatnonzero(reduced[rank:, column])
        if pivot_candidates.size == 0:
            continue

        pivot = rank + int(pivot_candidates[0])
        if pivot != rank:
            reduced[[rank, pivot]] = reduced[[pivot, rank]]

        for row in range(rows):
            if row != rank and reduced[row, column]:
                reduced[row] ^= reduced[rank]

        rank += 1
        if rank == rows:
            break

    return rank


def slice_rank_upper_bound(tensor: ArrayLike) -> int:
    """Return the sum of GF(2) ranks of the tensor's first-axis slices."""
    values = np.asarray(tensor)
    if values.ndim != 3:
        raise ValueError("tensor must be three-dimensional")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("a GF(2) tensor may contain only 0 and 1")
    return sum(gf2_matrix_rank(slice_) for slice_ in values)


@dataclass(frozen=True)
class StepResult:
    """Result returned after one TensorGame action."""

    observation: GF2Tensor
    reward: int
    terminated: bool
    truncated: bool
    info: dict[str, int | bool]


class TensorGame:
    """Find a short rank-one decomposition of the 2 x 2 product tensor.

    Each action is a triplet ``(u, v, w)``. Applying it XORs the rank-one
    tensor ``u outer v outer w`` from the current residual. Every move costs
    one point. A nonzero residual at the move limit receives an additional
    slice-rank upper-bound penalty.
    """

    def __init__(self, max_steps: int = 7) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self._target = matrix_multiplication_tensor(2)
        self._state = self._target.copy()
        self._actions: list[Factor] = []
        self._done = False

    @property
    def state(self) -> GF2Tensor:
        """Return a copy of the residual tensor."""
        return self._state.copy()

    @property
    def step_count(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> tuple[Factor, ...]:
        return tuple(self._actions)

    def reset(self) -> GF2Tensor:
        """Start a new game and return the initial multiplication tensor."""
        self._state = self._target.copy()
        self._actions.clear()
        self._done = False
        return self.state

    def step(self, action: Factor) -> StepResult:
        """Apply one rank-one factor to the residual tensor."""
        if self._done:
            raise RuntimeError("the game is over; call reset() before stepping")

        try:
            u, v, w = action
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a triplet (u, v, w)") from error

        term = rank_one_tensor(u, v, w)
        if term.shape != self._state.shape:
            raise ValueError("each action vector must have length 4")
        if not np.any(term):
            raise ValueError("action vectors must all be nonzero")

        self._state ^= term
        self._actions.append(action)

        terminated = not np.any(self._state)
        truncated = self.step_count >= self.max_steps and not terminated
        reward = -1
        rank_bound = 0

        if truncated:
            rank_bound = slice_rank_upper_bound(self._state)
            reward -= rank_bound

        self._done = terminated or truncated
        info: dict[str, int | bool] = {
            "step_count": self.step_count,
            "residual_nonzero": int(np.count_nonzero(self._state)),
            "rank_upper_bound": rank_bound,
            "success": terminated,
        }
        return StepResult(self.state, reward, terminated, truncated, info)

    def step_index(self, action_index: int, action_space: object) -> StepResult:
        """Apply a discrete action index from a compatible action catalog."""
        action_method = getattr(action_space, "action", None)
        if not callable(action_method):
            raise TypeError("action_space must provide an action(index) method")
        return self.step(action_method(action_index))
