import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphatensor.actions import ACTION_COUNT, GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.mcts import policy_guided_batched_mcts, policy_guided_mcts
from alphatensor.replay import TargetReplayBuffer
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


class ScriptedPolicyValueModel(torch.nn.Module):
    def __init__(self, target, action_space, actions):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        residual = target.copy()
        self.next_actions = {}
        for action in actions:
            index = action_space.index(action)
            self.next_actions[residual.tobytes()] = index
            residual ^= action_space.term(index)

    def forward(self, states):
        binary = ((states + 1.0) / 2.0).round().to(torch.uint8).cpu().numpy()
        logits = torch.full((len(states), ACTION_COUNT), -1000.0, device=states.device)
        for row, state in enumerate(binary):
            logits[row, self.next_actions[state.tobytes()]] = 1000.0
        return logits + self.anchor, torch.ones(len(states), device=states.device)


def test_scripted_mcts_finds_exact_rank_seven_decomposition() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = ScriptedPolicyValueModel(target, action_space, STRASSEN_GF2)
    result = policy_guided_mcts(
        model,
        target,
        action_space=action_space,
        simulations=12,
        top_k=1,
        root_noise_fraction=0.0,
        replay_trajectory_count=4,
        policy_target_count=4,
    )
    assert result.success
    assert len(result.action_indices) == 7
    assert verify_decomposition(
        target, [action_space.action(index) for index in result.action_indices]
    )
    assert not np.any(result.residual)
    assert len(result.trajectories) == 1
    assert result.trajectories[0].residual_rank_upper_bound == 0
    replay = TargetReplayBuffer(capacity=100)
    changed = replay.add_trajectory(
        target, result.trajectories[0], action_space=action_space
    )
    assert changed == 7
    assert result.policy_targets
    root_target = result.policy_targets[0]
    assert sum(root_target.visit_counts) > 0
    assert len(root_target.action_indices) == len(root_target.visit_counts)


def test_mcts_rejects_invalid_configuration() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = ScriptedPolicyValueModel(target, action_space, STRASSEN_GF2)
    with pytest.raises(ValueError, match="simulations"):
        policy_guided_mcts(model, target, simulations=0)
    with pytest.raises(ValueError, match="top_k"):
        policy_guided_mcts(model, target, top_k=ACTION_COUNT + 1)


def test_batched_mcts_finds_scripted_solution() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = ScriptedPolicyValueModel(target, action_space, STRASSEN_GF2)
    results = policy_guided_batched_mcts(
        model,
        target,
        action_space=action_space,
        num_trees=3,
        simulations_per_tree=12,
        top_k=1,
        root_noise_fraction=0.0,
    )
    assert results[0].success
    assert verify_decomposition(
        target, [action_space.action(index) for index in results[0].action_indices]
    )
