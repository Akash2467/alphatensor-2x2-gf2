import numpy as np
import pytest

from alphatensor.actions import (
    ACTION_COUNT,
    GF2ActionSpace,
    code_from_vector,
    vector_from_code,
)
from alphatensor.game import TensorGame
from alphatensor.tensor import rank_one_tensor


STRASSEN_FIRST_ACTION = (
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
)


def test_all_nonzero_vectors_round_trip() -> None:
    vectors = [vector_from_code(code) for code in range(1, 16)]

    assert len({tuple(vector) for vector in vectors}) == 15
    assert all(np.any(vector) for vector in vectors)
    assert [code_from_vector(vector) for vector in vectors] == list(range(1, 16))


def test_action_space_contains_3375_unique_actions() -> None:
    action_space = GF2ActionSpace()

    assert action_space.size == ACTION_COUNT == 3375
    assert action_space.terms.shape == (3375, 4, 4, 4)
    assert len(
        {action_space.index(action_space.action(i)) for i in range(3375)}
    ) == 3375


@pytest.mark.parametrize("index", [0, 1, 224, 225, 3374])
def test_action_indices_round_trip(index: int) -> None:
    action_space = GF2ActionSpace()

    action = action_space.action(index)

    assert action_space.index(action) == index
    np.testing.assert_array_equal(action_space.term(index), rank_one_tensor(*action))


def test_tensor_game_accepts_discrete_action_index() -> None:
    action_space = GF2ActionSpace()
    game = TensorGame(max_steps=7)
    index = action_space.index(STRASSEN_FIRST_ACTION)

    indexed_result = game.step_index(index, action_space)

    direct_game = TensorGame(max_steps=7)
    direct_result = direct_game.step(STRASSEN_FIRST_ACTION)
    np.testing.assert_array_equal(indexed_result.observation, direct_result.observation)


def test_invalid_vector_and_index_are_rejected() -> None:
    action_space = GF2ActionSpace()

    with pytest.raises(ValueError, match="nonzero"):
        code_from_vector([0, 0, 0, 0])
    with pytest.raises(IndexError):
        action_space.action(ACTION_COUNT)
