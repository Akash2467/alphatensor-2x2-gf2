import numpy as np
import pytest

from alphatensor.actions import GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.repair import local_repair_decomposition
from alphatensor.tensor import matrix_multiplication_tensor


STRASSEN_GF2 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
    ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]),
    ([0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
    ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def repairable_eight_actions(action_space: GF2ActionSpace) -> list[int]:
    u, v, _ = STRASSEN_GF2[0]
    split_one = (u, v, [1, 0, 0, 0])
    split_two = (u, v, [0, 0, 0, 1])
    return [
        action_space.index(split_one),
        action_space.index(split_two),
        *(action_space.index(factor) for factor in STRASSEN_GF2[1:]),
    ]


def test_local_repair_shortens_constructed_exact_decomposition() -> None:
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    actions = repairable_eight_actions(action_space)
    assert verify_decomposition(target, [action_space.action(i) for i in actions])

    result = local_repair_decomposition(
        target, actions, max_removed=2, action_space=action_space
    )

    assert result.success
    assert len(result.repaired_actions) == 7
    assert verify_decomposition(
        target, [action_space.action(i) for i in result.repaired_actions]
    )


def test_local_repair_rejects_nonexact_input() -> None:
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    with pytest.raises(ValueError, match="not an exact decomposition"):
        local_repair_decomposition(target, [0, 1], action_space=action_space)


def test_exact_five_to_four_meet_in_the_middle_repair() -> None:
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    actions = repairable_eight_actions(action_space)
    result = local_repair_decomposition(
        target,
        actions,
        min_removed=5,
        max_removed=5,
        action_space=action_space,
    )
    assert result.success
    assert len(result.removed_actions) == 5
    assert len(result.replacement_actions) == 4
    assert len(result.repaired_actions) == 7
    assert verify_decomposition(
        target, [action_space.action(index) for index in result.repaired_actions]
    )
