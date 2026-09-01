"""Quality scoring and checkpoint selection for target-search training."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class CheckpointQuality:
    search_success: bool
    residual_rank_upper_bound: int
    residual_nonzero: int
    validation_top5: float

    def key(self) -> tuple[int, int, int, float]:
        """Return a lexicographic key where larger values are better."""
        return (
            int(self.search_success),
            -self.residual_rank_upper_bound,
            -self.residual_nonzero,
            self.validation_top5,
        )


def is_better_checkpoint(
    candidate: CheckpointQuality, incumbent: CheckpointQuality
) -> bool:
    return candidate.key() > incumbent.key()


def outcome_weight(
    *,
    success: bool,
    residual_rank_upper_bound: int,
    residual_nonzero: int,
    temperature: float = 0.5,
) -> float:
    """Map terminal search quality to a positive replay weight."""
    if residual_rank_upper_bound < 0 or residual_nonzero < 0:
        raise ValueError("residual measures must be nonnegative")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if success:
        return 1.0
    penalty = residual_rank_upper_bound + residual_nonzero / 64.0
    return exp(-temperature * penalty)
