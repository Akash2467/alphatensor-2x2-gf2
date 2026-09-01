import numpy as np
import pytest

from alphatensor.actions import GF2ActionSpace
from alphatensor.expert import generate_expert_policy_targets
from alphatensor.tensor import matrix_multiplication_tensor


SOLUTION = (24, 332, 1117, 1646, 2479, 2888, 3315)


def test_expert_targets_cover_every_nonterminal_subset() -> None:
    catalog = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    targets = generate_expert_policy_targets(
        target, SOLUTION, action_space=catalog
    )

    assert len(targets) == 2**7 - 1
    assert targets[0].action_indices == SOLUTION
    np.testing.assert_array_equal(targets[0].state, target)
    assert len({item.state.tobytes() for item in targets}) == len(targets)

    for item in targets:
        assert item.remaining_steps == len(item.action_indices)
        assert item.visit_counts == (1,) * len(item.action_indices)
        reconstructed = np.zeros_like(item.state)
        for action in item.action_indices:
            reconstructed ^= catalog.term(action)
        np.testing.assert_array_equal(reconstructed, item.state)


def test_expert_targets_reject_an_inexact_decomposition() -> None:
    with pytest.raises(ValueError, match="do not exactly decompose"):
        generate_expert_policy_targets(
            matrix_multiplication_tensor(2), SOLUTION[:-1]
        )


def test_expert_targets_reject_duplicate_actions() -> None:
    with pytest.raises(ValueError, match="distinct"):
        generate_expert_policy_targets(
            matrix_multiplication_tensor(2), SOLUTION + (SOLUTION[0],)
        )
