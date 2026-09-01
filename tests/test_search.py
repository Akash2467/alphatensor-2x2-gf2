import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphatensor.actions import ACTION_COUNT, GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.search import policy_guided_beam_search
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
        binary_states = ((states + 1.0) / 2.0).round().to(torch.uint8).cpu().numpy()
        logits = torch.full((len(states), ACTION_COUNT), -1000.0, device=states.device)
        values = torch.zeros(len(states), device=states.device)
        for row, state in enumerate(binary_states):
            logits[row, self.next_actions[state.tobytes()]] = 1000.0
        return logits + self.anchor, values + self.anchor


class FiveCorrectMovesThenWrongModel(ScriptedPolicyValueModel):
    def __init__(self, target, action_space, actions):
        super().__init__(target, action_space, actions[:5])
        self.fallback = next(
            index
            for index in range(ACTION_COUNT)
            if index not in {action_space.index(action) for action in actions}
        )

    def forward(self, states):
        binary_states = ((states + 1.0) / 2.0).round().to(torch.uint8).cpu().numpy()
        logits = torch.full((len(states), ACTION_COUNT), -1000.0, device=states.device)
        for row, state in enumerate(binary_states):
            logits[row, self.next_actions.get(state.tobytes(), self.fallback)] = 1000.0
        return logits + self.anchor, torch.zeros(len(states), device=states.device)


def test_scripted_policy_finds_exact_rank_seven_decomposition() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = ScriptedPolicyValueModel(target, action_space, STRASSEN_GF2)

    result = policy_guided_beam_search(
        model,
        target,
        action_space=action_space,
        max_steps=7,
        beam_width=1,
        top_k=1,
    )
    factors = [action_space.action(index) for index in result.action_indices]

    assert result.success
    assert result.depth == 7
    assert verify_decomposition(target, factors)
    assert not np.any(result.residual)


def test_search_rejects_invalid_configuration() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = ScriptedPolicyValueModel(target, action_space, STRASSEN_GF2)

    with pytest.raises(ValueError, match="beam_width"):
        policy_guided_beam_search(model, target, beam_width=0)
    with pytest.raises(ValueError, match="top_k"):
        policy_guided_beam_search(model, target, top_k=ACTION_COUNT + 1)


def test_exact_tail_completes_last_two_moves() -> None:
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    model = FiveCorrectMovesThenWrongModel(target, action_space, STRASSEN_GF2)

    result = policy_guided_beam_search(
        model,
        target,
        action_space=action_space,
        max_steps=7,
        beam_width=1,
        top_k=1,
        exact_tail_steps=2,
    )

    factors = [action_space.action(index) for index in result.action_indices]
    assert result.success
    assert len(result.action_indices) == 7
    assert verify_decomposition(target, factors)
