import pytest

from alphatensor.selection import (
    CheckpointQuality,
    is_better_checkpoint,
    outcome_weight,
)


def test_checkpoint_selection_prioritizes_exact_search() -> None:
    incumbent = CheckpointQuality(False, 1, 1, 0.40)
    candidate = CheckpointQuality(True, 0, 0, 0.10)
    assert is_better_checkpoint(candidate, incumbent)


def test_checkpoint_selection_uses_validation_after_equal_search_quality() -> None:
    incumbent = CheckpointQuality(False, 1, 1, 0.29)
    candidate = CheckpointQuality(False, 1, 1, 0.30)
    assert is_better_checkpoint(candidate, incumbent)


def test_outcome_weight_rewards_better_terminal_results() -> None:
    exact = outcome_weight(success=True, residual_rank_upper_bound=0, residual_nonzero=0)
    near = outcome_weight(success=False, residual_rank_upper_bound=1, residual_nonzero=1)
    weak = outcome_weight(success=False, residual_rank_upper_bound=3, residual_nonzero=8)
    assert exact > near > weak > 0
    with pytest.raises(ValueError, match="temperature"):
        outcome_weight(
            success=False,
            residual_rank_upper_bound=1,
            residual_nonzero=1,
            temperature=-1,
        )
