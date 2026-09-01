import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphatensor.actions import GF2ActionSpace
from alphatensor.model import make_mixed_data_loader
from alphatensor.replay import TargetReplayBuffer
from alphatensor.search import SearchTrajectory
from alphatensor.synthetic import generate_synthetic_dataset
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


def exact_strassen_trajectory(action_space):
    return SearchTrajectory(
        action_indices=tuple(action_space.index(action) for action in STRASSEN_GF2),
        residual=np.zeros((4, 4, 4), dtype=np.uint8),
        score=0.0,
        residual_nonzero=0,
        residual_rank_upper_bound=0,
    )


def test_replay_buffer_records_exact_trajectory() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    buffer = TargetReplayBuffer(capacity=100)

    changed = buffer.add_trajectory(
        target, exact_strassen_trajectory(action_space), action_space=action_space
    )
    arrays = buffer.arrays()

    assert changed == len(buffer) == 7
    assert arrays.states.shape == (7, 4, 4, 4)
    np.testing.assert_array_equal(np.sort(arrays.remaining_steps), np.arange(1, 8))


def test_replay_buffer_save_load_round_trip(tmp_path) -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    buffer = TargetReplayBuffer(capacity=100)
    buffer.add_trajectory(
        target, exact_strassen_trajectory(action_space), action_space=action_space
    )
    path = tmp_path / "replay.npz"
    buffer.save(path)
    loaded = TargetReplayBuffer.load(path)

    np.testing.assert_array_equal(loaded.arrays().states, buffer.arrays().states)
    np.testing.assert_array_equal(
        loaded.arrays().action_indices, buffer.arrays().action_indices
    )


def test_replay_keeps_multiple_good_actions_for_the_same_state() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    original = exact_strassen_trajectory(action_space)
    reversed_trajectory = SearchTrajectory(
        action_indices=tuple(reversed(original.action_indices)),
        residual=original.residual,
        score=original.score,
        residual_nonzero=0,
        residual_rank_upper_bound=0,
    )
    buffer = TargetReplayBuffer(capacity=100, max_actions_per_state=4)
    buffer.add_trajectory(target, original, action_space=action_space)
    buffer.add_trajectory(target, reversed_trajectory, action_space=action_space)

    arrays = buffer.arrays()
    initial_actions = arrays.action_indices[
        np.all(arrays.states == target, axis=(1, 2, 3))
    ]
    assert set(initial_actions) == {
        original.action_indices[0],
        reversed_trajectory.action_indices[0],
    }


def test_mixed_loader_contains_requested_training_shapes() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    buffer = TargetReplayBuffer(capacity=100)
    buffer.add_trajectory(
        target, exact_strassen_trajectory(action_space), action_space=action_space
    )
    synthetic = generate_synthetic_dataset(10, ranks=3, seed=4)
    loader = make_mixed_data_loader(
        synthetic, buffer.arrays(), replay_fraction=0.25, batch_size=8, seed=4
    )

    states, actions, remaining = next(iter(loader))
    assert states.shape == (8, 4, 4, 4)
    assert actions.shape == remaining.shape == (8,)
