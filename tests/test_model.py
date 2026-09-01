import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphatensor.actions import ACTION_COUNT
from alphatensor.model import (
    PolicyValueNetwork,
    TensorTransformerPolicyValueNetwork,
    make_data_loader,
    policy_value_loss,
    predict_actions,
    run_epoch,
    sparse_policy_value_loss,
)
from alphatensor.synthetic import generate_synthetic_dataset


def test_network_output_shapes_and_backward_pass() -> None:
    model = PolicyValueNetwork(hidden_size=32, num_hidden_layers=2)
    states = torch.zeros((3, 4, 4, 4))
    actions = torch.tensor([0, 1, 2])
    remaining = torch.tensor([3.0, 2.0, 1.0])

    logits, predicted_remaining = model(states)
    losses = policy_value_loss(logits, predicted_remaining, actions, remaining)
    losses.total.backward()

    assert logits.shape == (3, ACTION_COUNT)
    assert predicted_remaining.shape == (3,)
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_transformer_output_shapes_and_backward_pass() -> None:
    model = TensorTransformerPolicyValueNetwork(
        d_model=32,
        num_heads=4,
        num_layers=1,
        feedforward_size=64,
        dropout=0.0,
    )
    states = torch.zeros((3, 4, 4, 4))
    actions = torch.tensor([0, 1, 2])
    remaining = torch.tensor([3.0, 2.0, 1.0])

    logits, predicted_remaining = model(states)
    losses = policy_value_loss(logits, predicted_remaining, actions, remaining)
    losses.total.backward()

    assert logits.shape == (3, ACTION_COUNT)
    assert predicted_remaining.shape == (3,)
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_sparse_policy_distribution_loss_backward_pass() -> None:
    model = PolicyValueNetwork(hidden_size=32, num_hidden_layers=1)
    states = torch.zeros((2, 4, 4, 4))
    actions = torch.tensor([[1, 3, -1], [2, 4, 8]])
    probabilities = torch.tensor([[0.75, 0.25, 0.0], [0.5, 0.3, 0.2]])
    remaining = torch.tensor([7.0, 5.0])
    logits, predicted = model(states)
    losses = sparse_policy_value_loss(
        logits, predicted, actions, probabilities, remaining
    )
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_transformer_uses_global_residual_projection() -> None:
    torch.manual_seed(11)
    model = TensorTransformerPolicyValueNetwork(
        d_model=16,
        num_heads=4,
        num_layers=1,
        feedforward_size=32,
        dropout=0.0,
    )
    model.eval()

    zero_state = torch.zeros((1, 4, 4, 4))
    changed_state = zero_state.clone()
    changed_state[0, 0, 0, 0] = 1
    zero_logits, zero_value = model(zero_state)
    changed_logits, changed_value = model(changed_state)

    assert not torch.equal(zero_logits, changed_logits)
    assert not torch.equal(zero_value, changed_value)


def test_training_epoch_and_prediction() -> None:
    torch.manual_seed(5)
    demonstrations = generate_synthetic_dataset(12, ranks=[3, 4], seed=5)
    loader = make_data_loader(demonstrations, batch_size=8, seed=5)
    model = PolicyValueNetwork(hidden_size=32, num_hidden_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = run_epoch(model, loader, optimizer=optimizer)
    state = torch.from_numpy(demonstrations.states[0] * 2.0 - 1.0).float()
    indices, probabilities, predicted_moves = predict_actions(model, state, top_k=5)

    assert metrics.examples == len(demonstrations)
    assert np.isfinite(metrics.loss)
    assert 0.0 <= metrics.policy_accuracy <= metrics.policy_top5_accuracy <= 1.0
    assert indices.shape == probabilities.shape == (5,)
    assert torch.all(probabilities >= 0)
    assert np.isfinite(predicted_moves)
