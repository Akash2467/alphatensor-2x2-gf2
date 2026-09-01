import numpy as np
import pytest

from alphatensor.game import TensorGame, gf2_matrix_rank
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


def test_reset_returns_initial_tensor() -> None:
    game = TensorGame()

    np.testing.assert_array_equal(game.reset(), matrix_multiplication_tensor(2))
    assert game.step_count == 0


def test_strassen_actions_win_in_seven_steps() -> None:
    game = TensorGame(max_steps=7)

    for index, action in enumerate(STRASSEN_GF2):
        result = game.step(action)
        assert result.reward == -1
        assert result.terminated is (index == 6)
        assert not result.truncated

    assert result.info["success"] is True
    assert result.info["residual_nonzero"] == 0
    assert game.step_count == 7
    assert not np.any(game.state)


def test_move_limit_truncates_and_penalizes_failure() -> None:
    game = TensorGame(max_steps=1)

    result = game.step(STRASSEN_GF2[0])

    assert not result.terminated
    assert result.truncated
    assert result.info["rank_upper_bound"] > 0
    assert result.reward < -1


def test_cannot_step_after_game_ends() -> None:
    game = TensorGame(max_steps=1)
    game.step(STRASSEN_GF2[0])

    with pytest.raises(RuntimeError, match="game is over"):
        game.step(STRASSEN_GF2[1])


def test_gf2_rank_differs_from_real_rank_when_required() -> None:
    matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    assert gf2_matrix_rank(matrix) == 2
