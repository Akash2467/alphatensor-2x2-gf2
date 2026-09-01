"""Synthetic demonstrations for supervised AlphaTensor pretraining."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .actions import ACTION_COUNT, GF2ActionSpace
from .tensor import GF2Tensor


@dataclass(frozen=True)
class SyntheticGame:
    """One exactly solvable tensor-decomposition trajectory."""

    target: GF2Tensor
    states: GF2Tensor
    action_indices: NDArray[np.int64]
    remaining_steps: NDArray[np.int64]

    @property
    def rank(self) -> int:
        return int(self.action_indices.size)


@dataclass(frozen=True)
class SyntheticDataset:
    """Flattened training examples collected from multiple synthetic games."""

    states: GF2Tensor
    action_indices: NDArray[np.int64]
    remaining_steps: NDArray[np.int64]
    game_ids: NDArray[np.int64]
    targets: GF2Tensor
    game_offsets: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.action_indices.size)

    @property
    def num_games(self) -> int:
        return int(self.targets.shape[0])

    def game_slice(self, game_id: int) -> slice:
        """Return the dataset slice containing one game's trajectory."""
        if not 0 <= game_id < self.num_games:
            raise IndexError("game_id is out of range")
        return slice(
            int(self.game_offsets[game_id]),
            int(self.game_offsets[game_id + 1]),
        )

    def save(self, path: str | Path) -> None:
        """Save the dataset as a compressed NumPy archive."""
        np.savez_compressed(
            Path(path),
            states=self.states,
            action_indices=self.action_indices,
            remaining_steps=self.remaining_steps,
            game_ids=self.game_ids,
            targets=self.targets,
            game_offsets=self.game_offsets,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticDataset":
        """Load a dataset created by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                states=archive["states"],
                action_indices=archive["action_indices"],
                remaining_steps=archive["remaining_steps"],
                game_ids=archive["game_ids"],
                targets=archive["targets"],
                game_offsets=archive["game_offsets"],
            )


def generate_synthetic_game(
    rank: int,
    *,
    action_space: GF2ActionSpace | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int = 100,
) -> SyntheticGame:
    """Generate an exact rank-one-factor trajectory of the requested length.

    Distinct actions are sampled without replacement. Games whose residual
    reaches zero before the final move are rejected because they contain a
    redundant suffix and therefore provide misleading remaining-step labels.
    """
    if not 1 <= rank <= ACTION_COUNT:
        raise ValueError(f"rank must be between 1 and {ACTION_COUNT}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    catalog = action_space or GF2ActionSpace()
    generator = rng or np.random.default_rng()

    for _ in range(max_attempts):
        actions = generator.choice(catalog.size, size=rank, replace=False).astype(
            np.int64
        )
        terms = np.stack([catalog.term(int(index)) for index in actions])
        target = np.bitwise_xor.reduce(terms, axis=0)

        states = np.empty((rank, 4, 4, 4), dtype=np.uint8)
        residual = target.copy()
        valid = True
        for step, term in enumerate(terms):
            if not np.any(residual):
                valid = False
                break
            states[step] = residual
            residual ^= term

        if valid and not np.any(residual):
            return SyntheticGame(
                target=target,
                states=states,
                action_indices=actions,
                remaining_steps=np.arange(rank, 0, -1, dtype=np.int64),
            )

    raise RuntimeError(
        f"could not generate a non-redundant rank-{rank} game in "
        f"{max_attempts} attempts"
    )


def generate_synthetic_dataset(
    num_games: int,
    ranks: int | Sequence[int] = 7,
    *,
    seed: int | None = None,
    action_space: GF2ActionSpace | None = None,
) -> SyntheticDataset:
    """Generate a reproducible collection of supervised training examples."""
    if num_games < 1:
        raise ValueError("num_games must be positive")

    rank_choices = np.atleast_1d(np.asarray(ranks, dtype=np.int64))
    if rank_choices.ndim != 1 or rank_choices.size == 0:
        raise ValueError("ranks must be an integer or a nonempty sequence")
    if np.any((rank_choices < 1) | (rank_choices > ACTION_COUNT)):
        raise ValueError(f"every rank must be between 1 and {ACTION_COUNT}")

    catalog = action_space or GF2ActionSpace()
    rng = np.random.default_rng(seed)
    games: list[SyntheticGame] = []
    for _ in range(num_games):
        rank = int(rng.choice(rank_choices))
        games.append(generate_synthetic_game(rank, action_space=catalog, rng=rng))

    lengths = np.array([game.rank for game in games], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)))
    return SyntheticDataset(
        states=np.concatenate([game.states for game in games]),
        action_indices=np.concatenate([game.action_indices for game in games]),
        remaining_steps=np.concatenate([game.remaining_steps for game in games]),
        game_ids=np.repeat(np.arange(num_games, dtype=np.int64), lengths),
        targets=np.stack([game.target for game in games]),
        game_offsets=offsets,
    )
