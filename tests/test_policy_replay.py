import numpy as np
import pytest

from alphatensor.mcts import MCTSPolicyTarget
from alphatensor.policy_replay import PolicyTargetReplayBuffer
from alphatensor.tensor import matrix_multiplication_tensor


def test_policy_replay_merges_visit_distributions() -> None:
    state = matrix_multiplication_tensor(2)
    buffer = PolicyTargetReplayBuffer(capacity=10, max_actions_per_state=4)
    buffer.add(MCTSPolicyTarget(state, (2, 7), (3, 1), 7.0))
    buffer.add(MCTSPolicyTarget(state, (2, 9), (1, 4), 7.0))
    arrays = buffer.arrays()

    assert len(buffer) == 1
    assert set(arrays.action_indices[0, :3]) == {2, 7, 9}
    np.testing.assert_allclose(arrays.probabilities[0].sum(), 1.0)


def test_policy_replay_save_load_round_trip(tmp_path) -> None:
    state = matrix_multiplication_tensor(2)
    buffer = PolicyTargetReplayBuffer(capacity=10, max_actions_per_state=4)
    buffer.add(MCTSPolicyTarget(state, (4, 5), (5, 2), 7.0))
    path = tmp_path / "policy_replay.npz"
    buffer.save(path)
    loaded = PolicyTargetReplayBuffer.load(path)
    np.testing.assert_array_equal(
        loaded.arrays().action_indices, buffer.arrays().action_indices
    )
    np.testing.assert_allclose(
        loaded.arrays().probabilities, buffer.arrays().probabilities
    )


def test_policy_replay_applies_outcome_weight() -> None:
    first = matrix_multiplication_tensor(2)
    second = first.copy()
    second[0, 0, 0] ^= 1
    buffer = PolicyTargetReplayBuffer(capacity=10, max_actions_per_state=4)
    buffer.add(MCTSPolicyTarget(first, (1,), (10,), 7.0), weight=1.0)
    buffer.add(MCTSPolicyTarget(second, (2,), (10,), 7.0), weight=0.25)
    arrays = buffer.arrays()
    assert arrays.total_visits.max() == pytest.approx(10.0)
    assert arrays.total_visits.min() == pytest.approx(2.5)
