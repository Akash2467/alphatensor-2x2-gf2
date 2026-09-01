"""Expert policy targets derived from exact tensor decompositions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .actions import GF2ActionSpace
from .decomposition import verify_decomposition
from .mcts import MCTSPolicyTarget


def generate_expert_policy_targets(
    target: ArrayLike,
    action_indices: Iterable[int],
    *,
    action_space: GF2ActionSpace | None = None,
) -> tuple[MCTSPolicyTarget, ...]:
    """Return one soft policy target for every nonterminal subset state.

    For a decomposition with ``r`` distinct factors this creates ``2**r - 1``
    states.  At each state every unapplied factor is a correct next action, so
    the targets cover every ordering of the commutative GF(2) decomposition.
    """
    catalog = action_space or GF2ActionSpace()
    state = np.asarray(target, dtype=np.uint8)
    if state.shape != (4, 4, 4):
        raise ValueError("target must have shape (4, 4, 4)")
    if np.any((state != 0) & (state != 1)):
        raise ValueError("target must contain only GF(2) values")

    actions = tuple(int(index) for index in action_indices)
    if not actions:
        raise ValueError("action_indices must be nonempty")
    if len(set(actions)) != len(actions):
        raise ValueError("action_indices must be distinct")
    factors = tuple(catalog.action(index) for index in actions)
    if not verify_decomposition(state, factors):
        raise ValueError("action_indices do not exactly decompose target")

    terms = tuple(catalog.term(index) for index in actions)
    targets: list[MCTSPolicyTarget] = []
    complete_mask = (1 << len(actions)) - 1
    for applied_mask in range(complete_mask):
        residual = state.copy()
        remaining_actions: list[int] = []
        for position, (action, term) in enumerate(zip(actions, terms)):
            if applied_mask & (1 << position):
                residual ^= term
            else:
                remaining_actions.append(action)
        targets.append(
            MCTSPolicyTarget(
                state=residual,
                action_indices=tuple(remaining_actions),
                visit_counts=(1,) * len(remaining_actions),
                remaining_steps=float(len(remaining_actions)),
            )
        )
    return tuple(targets)


def load_exact_solutions(path: str | Path) -> tuple[tuple[int, ...], ...]:
    """Load exact solver action lists from a repeatability-results JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    solutions = []
    for run in payload.get("runs", []):
        if run.get("solver_exact"):
            solutions.append(tuple(int(index) for index in run["solver_action_indices"]))
    if not solutions:
        raise ValueError(f"no exact solver solutions found in {path}")
    return tuple(solutions)
