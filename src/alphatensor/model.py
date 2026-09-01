"""Policy-value neural network and supervised training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .actions import ACTION_COUNT
from .synthetic import SyntheticDataset


class TensorStateDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Expose synthetic demonstrations as PyTorch training samples."""

    def __init__(self, demonstrations: SyntheticDataset) -> None:
        self.states = torch.from_numpy(demonstrations.states).to(torch.float32)
        self.action_indices = torch.from_numpy(demonstrations.action_indices).long()
        self.remaining_steps = torch.from_numpy(
            demonstrations.remaining_steps
        ).to(torch.float32)

    def __len__(self) -> int:
        return int(self.action_indices.numel())

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        # Center binary inputs around zero to make optimization easier.
        state = self.states[index] * 2.0 - 1.0
        return state, self.action_indices[index], self.remaining_steps[index]


class SparsePolicyTargetDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Expose sparse MCTS visit distributions as soft policy targets."""

    def __init__(self, targets: object) -> None:
        self.states = torch.from_numpy(targets.states).to(torch.float32)  # type: ignore[attr-defined]
        self.action_indices = torch.from_numpy(targets.action_indices).long()  # type: ignore[attr-defined]
        self.probabilities = torch.from_numpy(targets.probabilities).to(torch.float32)  # type: ignore[attr-defined]
        self.remaining_steps = torch.from_numpy(targets.remaining_steps).to(torch.float32)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.states[index] * 2.0 - 1.0,
            self.action_indices[index],
            self.probabilities[index],
            self.remaining_steps[index],
        )


def make_mixed_data_loader(
    synthetic: SyntheticDataset,
    replay: object,
    *,
    replay_fraction: float = 0.25,
    batch_size: int = 128,
    seed: int = 0,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Sample a fixed mixture of synthetic and target-replay examples."""
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError("replay_fraction must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    synthetic_dataset = TensorStateDataset(synthetic)
    replay_dataset = TensorStateDataset(replay)  # type: ignore[arg-type]
    if len(replay_dataset) == 0:
        raise ValueError("replay dataset must not be empty")
    combined = ConcatDataset([synthetic_dataset, replay_dataset])
    synthetic_weight = (1.0 - replay_fraction) / len(synthetic_dataset)
    replay_weight = replay_fraction / len(replay_dataset)
    weights = torch.cat(
        (
            torch.full((len(synthetic_dataset),), synthetic_weight),
            torch.full((len(replay_dataset),), replay_weight),
        )
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(synthetic_dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(combined, batch_size=batch_size, sampler=sampler)


class PolicyValueNetwork(nn.Module):
    """Small MLP for the fixed 4 x 4 x 4 residual tensor.

    The policy head predicts one of all 3,375 discrete actions. The value head
    estimates how many decomposition moves remain from the current state.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_hidden_layers: int = 3,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
        if hidden_size < 2:
            raise ValueError("hidden_size must be at least 2")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        if action_count < 1:
            raise ValueError("action_count must be positive")

        layers: list[nn.Module] = []
        input_size = 4 * 4 * 4
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                ]
            )
            input_size = hidden_size

        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return policy logits and predicted remaining moves."""
        if states.ndim == 3:
            states = states.unsqueeze(0)
        if states.ndim != 4 or tuple(states.shape[-3:]) != (4, 4, 4):
            raise ValueError("states must have shape (batch, 4, 4, 4)")

        features = self.torso(states.to(torch.float32).flatten(start_dim=1))
        policy_logits = self.policy_head(features)
        remaining_moves = self.value_head(features).squeeze(-1)
        return policy_logits, remaining_moves


class TensorTransformerPolicyValueNetwork(nn.Module):
    """Transformer policy-value model for the 64 entries of a 4 x 4 x 4 state.

    Every tensor entry becomes a token. A learned classification token gathers
    global information through self-attention and feeds the policy/value heads.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_size: int = 256,
        dropout: float = 0.1,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be at least 2")
        if num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if feedforward_size < 1:
            raise ValueError("feedforward_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if action_count < 1:
            raise ValueError("action_count must be positive")

        self.input_projection = nn.Linear(1, d_model)
        self.global_projection = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
        )
        self.classification_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, 65, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, action_count)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.trunc_normal_(self.classification_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return policy logits and predicted remaining moves."""
        if states.ndim == 3:
            states = states.unsqueeze(0)
        if states.ndim != 4 or tuple(states.shape[-3:]) != (4, 4, 4):
            raise ValueError("states must have shape (batch, 4, 4, 4)")

        batch_size = states.shape[0]
        flat_states = states.to(torch.float32).reshape(batch_size, 64)
        global_features = self.global_projection(flat_states)
        tokens = flat_states.unsqueeze(-1)
        tokens = self.input_projection(tokens)
        classification_token = self.classification_token.expand(batch_size, -1, -1)
        tokens = torch.cat((classification_token, tokens), dim=1)
        tokens = tokens + self.position_embedding

        encoded = self.encoder(tokens)
        representation = self.output_norm(encoded[:, 0] + global_features)
        policy_logits = self.policy_head(representation)
        remaining_moves = self.value_head(representation).squeeze(-1)
        return policy_logits, remaining_moves


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    policy: Tensor
    value: Tensor


def policy_value_loss(
    policy_logits: Tensor,
    predicted_remaining: Tensor,
    target_actions: Tensor,
    target_remaining: Tensor,
    *,
    value_weight: float = 0.25,
) -> LossOutput:
    """Combine action cross-entropy with a remaining-moves regression loss."""
    if value_weight < 0:
        raise ValueError("value_weight must be nonnegative")
    policy_loss = nn.functional.cross_entropy(policy_logits, target_actions)
    value_loss = nn.functional.smooth_l1_loss(
        predicted_remaining, target_remaining.to(torch.float32)
    )
    return LossOutput(
        total=policy_loss + value_weight * value_loss,
        policy=policy_loss,
        value=value_loss,
    )


def sparse_policy_value_loss(
    policy_logits: Tensor,
    predicted_remaining: Tensor,
    target_actions: Tensor,
    target_probabilities: Tensor,
    target_remaining: Tensor,
    *,
    value_weight: float = 0.25,
) -> LossOutput:
    """Cross-entropy against a sparse MCTS visit-count distribution."""
    if value_weight < 0:
        raise ValueError("value_weight must be nonnegative")
    valid = target_actions >= 0
    safe_actions = target_actions.clamp_min(0)
    log_probabilities = torch.log_softmax(policy_logits, dim=1)
    selected = log_probabilities.gather(1, safe_actions)
    weights = target_probabilities * valid.to(target_probabilities.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    policy_loss = -(weights * selected).sum(dim=1).mean()
    value_loss = nn.functional.smooth_l1_loss(
        predicted_remaining, target_remaining.to(torch.float32)
    )
    return LossOutput(
        total=policy_loss + value_weight * value_loss,
        policy=policy_loss,
        value=value_loss,
    )


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    policy_accuracy: float
    policy_top5_accuracy: float
    examples: int


def make_data_loader(
    demonstrations: SyntheticDataset,
    *,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Construct a deterministic data loader for demonstration samples."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorStateDataset(demonstrations),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def make_sparse_policy_loader(
    targets: object,
    *,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SparsePolicyTargetDataset(targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def run_sparse_policy_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    value_weight: float = 0.25,
) -> EpochMetrics:
    """Train or evaluate using sparse MCTS policy distributions."""
    training = optimizer is not None
    model.train(training)
    target_device = torch.device(device)
    model.to(target_device)
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0}
    correct = top5_correct = examples = 0

    for states, actions, probabilities, remaining in data_loader:
        states = states.to(target_device)
        actions = actions.to(target_device)
        probabilities = probabilities.to(target_device)
        remaining = remaining.to(target_device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, predicted = model(states)
            losses = sparse_policy_value_loss(
                logits, predicted, actions, probabilities, remaining,
                value_weight=value_weight,
            )
            if training:
                losses.total.backward()
                optimizer.step()
        batch_size = int(states.shape[0])
        primary = actions.gather(1, probabilities.argmax(dim=1, keepdim=True)).squeeze(1)
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        valid_targets = actions >= 0
        any_target = top5.unsqueeze(2).eq(actions.unsqueeze(1)) & valid_targets.unsqueeze(1)
        correct += int((predictions == primary).sum())
        top5_correct += int(any_target.any(dim=2).any(dim=1).sum())
        totals["loss"] += float(losses.total.detach()) * batch_size
        totals["policy"] += float(losses.policy.detach()) * batch_size
        totals["value"] += float(losses.value.detach()) * batch_size
        examples += batch_size
    if examples == 0:
        raise ValueError("data_loader contains no examples")
    return EpochMetrics(
        loss=totals["loss"] / examples,
        policy_loss=totals["policy"] / examples,
        value_loss=totals["value"] / examples,
        policy_accuracy=correct / examples,
        policy_top5_accuracy=top5_correct / examples,
        examples=examples,
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    value_weight: float = 0.25,
) -> EpochMetrics:
    """Run one training epoch, or an evaluation epoch if optimizer is omitted."""
    training = optimizer is not None
    model.train(training)
    target_device = torch.device(device)
    model.to(target_device)

    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0}
    correct = 0
    top5_correct = 0
    examples = 0

    for states, actions, remaining in data_loader:
        states = states.to(target_device)
        actions = actions.to(target_device)
        remaining = remaining.to(target_device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, predicted_remaining = model(states)
            losses = policy_value_loss(
                logits,
                predicted_remaining,
                actions,
                remaining,
                value_weight=value_weight,
            )
            if training:
                losses.total.backward()
                optimizer.step()

        batch_size = int(actions.shape[0])
        totals["loss"] += float(losses.total.detach()) * batch_size
        totals["policy"] += float(losses.policy.detach()) * batch_size
        totals["value"] += float(losses.value.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == actions).sum())
        top5_indices = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top5_correct += int(top5_indices.eq(actions.unsqueeze(1)).any(dim=1).sum())
        examples += batch_size

    if examples == 0:
        raise ValueError("data_loader contains no examples")
    return EpochMetrics(
        loss=totals["loss"] / examples,
        policy_loss=totals["policy"] / examples,
        value_loss=totals["value"] / examples,
        policy_accuracy=correct / examples,
        policy_top5_accuracy=top5_correct / examples,
        examples=examples,
    )


@torch.no_grad()
def predict_actions(
    model: nn.Module,
    state: Tensor,
    *,
    top_k: int = 10,
) -> tuple[Tensor, Tensor, float]:
    """Return top action indices, probabilities, and predicted moves."""
    if not 1 <= top_k <= ACTION_COUNT:
        raise ValueError(f"top_k must be between 1 and {ACTION_COUNT}")
    model.eval()
    device = next(model.parameters()).device
    logits, remaining = model(state.to(device))
    probabilities = logits.softmax(dim=-1)
    top_probabilities, top_indices = probabilities.topk(top_k, dim=-1)
    return (
        top_indices[0].cpu(),
        top_probabilities[0].cpu(),
        float(remaining[0].cpu()),
    )


def load_policy_value_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct a policy-value model from a training checkpoint."""
    target_device = torch.device(device)
    checkpoint = torch.load(Path(path), map_location=target_device, weights_only=True)
    architecture = checkpoint.get("architecture", "mlp")

    if architecture == "transformer":
        model = TensorTransformerPolicyValueNetwork(**checkpoint["model_config"])
    elif architecture == "mlp":
        config = checkpoint.get("model_config")
        if config is None:
            config = {
                "hidden_size": checkpoint.get("hidden_size", 256),
                "num_hidden_layers": checkpoint.get("hidden_layers", 3),
            }
        model = PolicyValueNetwork(**config)
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint
