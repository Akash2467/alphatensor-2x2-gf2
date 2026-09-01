"""Target-specific replay data for AlphaTensor self-play fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .actions import GF2ActionSpace
from .search import SearchTrajectory
from .tensor import GF2Tensor


@dataclass(frozen=True)
class ReplayArrays:
    states: GF2Tensor
    action_indices: NDArray[np.int64]
    remaining_steps: NDArray[np.float32]
    qualities: NDArray[np.float32]
    trajectory_scores: NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.action_indices.size)


@dataclass(frozen=True)
class ReplayRecord:
    state: GF2Tensor
    action_index: int
    remaining_steps: float
    quality: float
    trajectory_score: float


class TargetReplayBuffer:
    """Bounded state-deduplicated replay buffer for target search trajectories."""

    def __init__(self, capacity: int = 50_000, max_actions_per_state: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if max_actions_per_state < 1:
            raise ValueError("max_actions_per_state must be positive")
        self.capacity = capacity
        self.max_actions_per_state = max_actions_per_state
        self._records: dict[tuple[bytes, int], ReplayRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add_trajectory(
        self,
        target: GF2Tensor,
        trajectory: SearchTrajectory,
        *,
        action_space: GF2ActionSpace | None = None,
    ) -> int:
        """Add all state/action pairs from a trajectory and return replacements."""
        catalog = action_space or GF2ActionSpace()
        residual = np.asarray(target, dtype=np.uint8).copy()
        if residual.shape != (4, 4, 4):
            raise ValueError("target must have shape (4, 4, 4)")

        terminal_penalty = trajectory.residual_rank_upper_bound
        trajectory_quality = float(len(trajectory.action_indices) + terminal_penalty)
        changed = 0
        for step, action_index in enumerate(trajectory.action_indices):
            remaining_steps = float(
                len(trajectory.action_indices) - step + terminal_penalty
            )
            record = ReplayRecord(
                state=residual.copy(),
                action_index=int(action_index),
                remaining_steps=remaining_steps,
                quality=trajectory_quality,
                trajectory_score=trajectory.score,
            )
            state_key = residual.tobytes()
            key = (state_key, int(action_index))
            previous = self._records.get(key)
            if previous is None or (
                record.quality < previous.quality
                or (
                    record.quality == previous.quality
                    and record.trajectory_score > previous.trajectory_score
                )
            ):
                self._records[key] = record
                changed += 1
            self._trim_state(state_key)
            residual ^= catalog.term(int(action_index))

        if not np.array_equal(residual, trajectory.residual):
            raise ValueError("trajectory actions do not reproduce its residual")
        self._trim()
        return changed

    def _trim_state(self, state_key: bytes) -> None:
        matching = [key for key in self._records if key[0] == state_key]
        overflow = len(matching) - self.max_actions_per_state
        if overflow <= 0:
            return
        worst = sorted(
            matching,
            key=lambda key: (
                self._records[key].quality,
                -self._records[key].trajectory_score,
            ),
            reverse=True,
        )[:overflow]
        for key in worst:
            del self._records[key]

    def _trim(self) -> None:
        overflow = len(self._records) - self.capacity
        if overflow <= 0:
            return
        worst_keys = sorted(
            self._records,
            key=lambda key: (
                self._records[key].quality,
                -self._records[key].trajectory_score,
            ),
            reverse=True,
        )[:overflow]
        for key in worst_keys:
            del self._records[key]

    def arrays(self) -> ReplayArrays:
        if not self._records:
            raise ValueError("replay buffer is empty")
        records = sorted(
            self._records.values(),
            key=lambda record: (record.quality, -record.trajectory_score),
        )
        return ReplayArrays(
            states=np.stack([record.state for record in records]),
            action_indices=np.array(
                [record.action_index for record in records], dtype=np.int64
            ),
            remaining_steps=np.array(
                [record.remaining_steps for record in records], dtype=np.float32
            ),
            qualities=np.array([record.quality for record in records], dtype=np.float32),
            trajectory_scores=np.array(
                [record.trajectory_score for record in records], dtype=np.float32
            ),
        )

    def save(self, path: str | Path) -> None:
        arrays = self.arrays()
        np.savez_compressed(
            Path(path),
            states=arrays.states,
            action_indices=arrays.action_indices,
            remaining_steps=arrays.remaining_steps,
            qualities=arrays.qualities,
            trajectory_scores=arrays.trajectory_scores,
            capacity=np.array(self.capacity, dtype=np.int64),
            max_actions_per_state=np.array(self.max_actions_per_state, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TargetReplayBuffer":
        with np.load(Path(path), allow_pickle=False) as archive:
            max_actions = int(archive["max_actions_per_state"]) if "max_actions_per_state" in archive else 1
            buffer = cls(
                capacity=int(archive["capacity"]),
                max_actions_per_state=max_actions,
            )
            scores = archive["trajectory_scores"] if "trajectory_scores" in archive else np.zeros_like(archive["qualities"])
            for state, action, remaining, quality, score in zip(
                archive["states"],
                archive["action_indices"],
                archive["remaining_steps"],
                archive["qualities"],
                scores,
            ):
                buffer._records[(state.tobytes(), int(action))] = ReplayRecord(
                    state=state.copy(),
                    action_index=int(action),
                    remaining_steps=float(remaining),
                    quality=float(quality),
                    trajectory_score=float(score),
                )
        return buffer
