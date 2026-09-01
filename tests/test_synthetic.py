import numpy as np

from alphatensor.actions import GF2ActionSpace
from alphatensor.synthetic import (
    SyntheticDataset,
    generate_synthetic_dataset,
    generate_synthetic_game,
)


def replay_game(states, actions, action_space):
    residual = states[0].copy()
    for step, action_index in enumerate(actions):
        np.testing.assert_array_equal(residual, states[step])
        residual ^= action_space.term(int(action_index))
    return residual


def test_synthetic_game_is_exactly_solvable() -> None:
    action_space = GF2ActionSpace()
    game = generate_synthetic_game(
        7, action_space=action_space, rng=np.random.default_rng(42)
    )

    assert game.states.shape == (7, 4, 4, 4)
    assert len(np.unique(game.action_indices)) == 7
    np.testing.assert_array_equal(game.states[0], game.target)
    np.testing.assert_array_equal(game.remaining_steps, [7, 6, 5, 4, 3, 2, 1])
    assert not np.any(replay_game(game.states, game.action_indices, action_space))


def test_dataset_is_reproducible_and_has_correct_labels() -> None:
    first = generate_synthetic_dataset(5, ranks=[5, 6, 7], seed=123)
    second = generate_synthetic_dataset(5, ranks=[5, 6, 7], seed=123)

    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.action_indices, second.action_indices)
    np.testing.assert_array_equal(first.remaining_steps, second.remaining_steps)
    assert first.num_games == 5
    assert len(first) == int(first.game_offsets[-1])


def test_every_dataset_game_reaches_zero() -> None:
    action_space = GF2ActionSpace()
    dataset = generate_synthetic_dataset(
        8, ranks=[3, 5, 7], seed=9, action_space=action_space
    )

    for game_id in range(dataset.num_games):
        game_slice = dataset.game_slice(game_id)
        states = dataset.states[game_slice]
        actions = dataset.action_indices[game_slice]
        assert not np.any(replay_game(states, actions, action_space))


def test_dataset_save_and_load(tmp_path) -> None:
    dataset = generate_synthetic_dataset(3, ranks=4, seed=7)
    path = tmp_path / "demonstrations.npz"
    dataset.save(path)
    loaded = SyntheticDataset.load(path)

    np.testing.assert_array_equal(loaded.states, dataset.states)
    np.testing.assert_array_equal(loaded.action_indices, dataset.action_indices)
    np.testing.assert_array_equal(loaded.game_offsets, dataset.game_offsets)
