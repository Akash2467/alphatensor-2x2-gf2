"""Sparse replay storage for MCTS visit-count policy distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .actions import GF2ActionSpace
from .mcts import MCTSPolicyTarget
from .symmetry import BasisTransform, transform_action_index, transform_tensor
from .tensor import GF2Tensor


@dataclass(frozen=True)
class SparsePolicyArrays:
    states: GF2Tensor
    action_indices: NDArray[np.int64]
    probabilities: NDArray[np.float32]
    remaining_steps: NDArray[np.float32]
    total_visits: NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.states.shape[0])


@dataclass
class _PolicyRecord:
    state: GF2Tensor
    counts: dict[int, float]
    remaining_steps: float
    total_visits: float


def transform_policy_target(
    target: MCTSPolicyTarget,
    transform: BasisTransform,
    *,
    action_space: GF2ActionSpace | None = None,
) -> MCTSPolicyTarget:
    catalog = action_space or GF2ActionSpace()
    return MCTSPolicyTarget(
        state=transform_tensor(target.state, transform),
        action_indices=tuple(
            transform_action_index(action, transform, action_space=catalog)
            for action in target.action_indices
        ),
        visit_counts=target.visit_counts,
        remaining_steps=target.remaining_steps,
    )


class PolicyTargetReplayBuffer:
    """Merge visit counts for repeated states into sparse policy targets."""

    def __init__(
        self, capacity: int = 50_000, max_actions_per_state: int = 16
    ) -> None:
        if capacity < 1 or max_actions_per_state < 1:
            raise ValueError("capacity and max_actions_per_state must be positive")
        self.capacity = capacity
        self.max_actions_per_state = max_actions_per_state
        self._records: dict[bytes, _PolicyRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add(self, target: MCTSPolicyTarget, *, weight: float = 1.0) -> bool:
        state = np.asarray(target.state, dtype=np.uint8)
        if state.shape != (4, 4, 4):
            raise ValueError("policy target state must have shape (4, 4, 4)")
        if len(target.action_indices) != len(target.visit_counts) or not target.action_indices:
            raise ValueError("actions and visit counts must be nonempty and aligned")
        if any(count <= 0 for count in target.visit_counts):
            raise ValueError("visit counts must be positive")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        key = state.tobytes()
        weighted_counts = [float(count) * weight for count in target.visit_counts]
        incoming_total = float(sum(weighted_counts))
        record = self._records.get(key)
        if record is None:
            record = _PolicyRecord(state.copy(), {}, target.remaining_steps, 0.0)
            self._records[key] = record
        combined_total = record.total_visits + incoming_total
        record.remaining_steps = (
            record.remaining_steps * record.total_visits
            + target.remaining_steps * incoming_total
        ) / combined_total
        record.total_visits = combined_total
        for action, count in zip(target.action_indices, weighted_counts):
            record.counts[int(action)] = record.counts.get(int(action), 0.0) + count
        if len(record.counts) > self.max_actions_per_state:
            record.counts = dict(
                sorted(record.counts.items(), key=lambda item: item[1], reverse=True)[
                    : self.max_actions_per_state
                ]
            )
        self._trim()
        return record.total_visits == incoming_total

    def _trim(self) -> None:
        overflow = len(self._records) - self.capacity
        if overflow <= 0:
            return
        worst = sorted(
            self._records, key=lambda key: self._records[key].total_visits
        )[:overflow]
        for key in worst:
            del self._records[key]

    def arrays(self) -> SparsePolicyArrays:
        if not self._records:
            raise ValueError("policy replay buffer is empty")
        records = sorted(
            self._records.values(), key=lambda record: record.total_visits, reverse=True
        )
        width = self.max_actions_per_state
        actions = np.full((len(records), width), -1, dtype=np.int64)
        probabilities = np.zeros((len(records), width), dtype=np.float32)
        for row, record in enumerate(records):
            ranked = sorted(record.counts.items(), key=lambda item: item[1], reverse=True)
            total = sum(count for _, count in ranked)
            for column, (action, count) in enumerate(ranked[:width]):
                actions[row, column] = action
                probabilities[row, column] = count / total
        return SparsePolicyArrays(
            states=np.stack([record.state for record in records]),
            action_indices=actions,
            probabilities=probabilities,
            remaining_steps=np.array(
                [record.remaining_steps for record in records], dtype=np.float32
            ),
            total_visits=np.array(
                [record.total_visits for record in records], dtype=np.float32
            ),
        )

    def save(self, path: str | Path) -> None:
        arrays = self.arrays()
        np.savez_compressed(
            Path(path),
            states=arrays.states,
            action_indices=arrays.action_indices,
            probabilities=arrays.probabilities,
            remaining_steps=arrays.remaining_steps,
            total_visits=arrays.total_visits,
            capacity=np.array(self.capacity, dtype=np.int64),
            max_actions_per_state=np.array(self.max_actions_per_state, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PolicyTargetReplayBuffer":
        with np.load(Path(path), allow_pickle=False) as archive:
            buffer = cls(
                capacity=int(archive["capacity"]),
                max_actions_per_state=int(archive["max_actions_per_state"]),
            )
            for state, actions, probabilities, remaining, total in zip(
                archive["states"],
                archive["action_indices"],
                archive["probabilities"],
                archive["remaining_steps"],
                archive["total_visits"],
            ):
                counts = {
                    int(action): float(probability * total)
                    for action, probability in zip(actions, probabilities)
                    if action >= 0 and probability > 0
                }
                buffer._records[state.tobytes()] = _PolicyRecord(
                    state.copy(), counts, float(remaining), float(total)
                )
        return buffer
