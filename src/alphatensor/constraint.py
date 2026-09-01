"""Global exact rank constraints over the finite GF(2) action catalog."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from .actions import GF2ActionSpace
from .decomposition import verify_decomposition


@dataclass(frozen=True)
class ConstraintSolveResult:
    status: str
    rank: int
    action_indices: tuple[int, ...]
    elapsed_seconds: float
    exact_verification: bool
    reason_unknown: str = ""


def _xor_all(z3: object, expressions: list[object]) -> object:
    """Build a balanced XOR tree for Z3 versions with fixed-arity Xor."""
    if not expressions:
        return z3.BoolVal(False)  # type: ignore[attr-defined]
    level = expressions
    while len(level) > 1:
        next_level = [
            z3.Xor(level[index], level[index + 1])  # type: ignore[attr-defined]
            for index in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def solve_exact_rank(
    target: ArrayLike,
    rank: int,
    *,
    action_space: GF2ActionSpace | None = None,
    timeout_seconds: float = 300.0,
    random_seed: int = 0,
) -> ConstraintSolveResult:
    """Solve for exactly ``rank`` distinct catalog actions using SMT parity."""
    try:
        import z3
    except ImportError as error:
        raise RuntimeError(
            "z3-solver is required; install the project's 'exact' extra"
        ) from error
    values = np.asarray(target)
    if values.shape != (4, 4, 4):
        raise ValueError("target must have shape (4, 4, 4)")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("target must contain only GF(2) values")
    catalog = action_space or GF2ActionSpace()
    if not 0 <= rank <= catalog.size:
        raise ValueError("rank is outside the action-space range")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    selected = [z3.Bool(f"action_{index}") for index in range(catalog.size)]
    solver = z3.SolverFor("QF_FD")
    solver.set(timeout=int(timeout_seconds * 1000), random_seed=random_seed)
    solver.add(z3.PbEq([(variable, 1) for variable in selected], rank))
    terms = catalog.terms.reshape(catalog.size, 64)
    flat_target = values.reshape(64)
    for coordinate in range(64):
        contributors = np.flatnonzero(terms[:, coordinate])
        parity = _xor_all(
            z3, [selected[int(index)] for index in contributors]
        )
        solver.add(parity if flat_target[coordinate] else z3.Not(parity))

    started = perf_counter()
    status = solver.check()
    elapsed = perf_counter() - started
    if status == z3.sat:
        model = solver.model()
        actions = tuple(
            index
            for index, variable in enumerate(selected)
            if z3.is_true(model.evaluate(variable, model_completion=True))
        )
        exact = verify_decomposition(
            values, [catalog.action(index) for index in actions]
        )
        return ConstraintSolveResult("sat", rank, actions, elapsed, exact)
    if status == z3.unsat:
        return ConstraintSolveResult("unsat", rank, (), elapsed, False)
    return ConstraintSolveResult(
        "unknown", rank, (), elapsed, False, solver.reason_unknown()
    )
