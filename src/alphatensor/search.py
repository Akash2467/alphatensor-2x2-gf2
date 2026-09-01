"""Policy-guided beam search for exact GF(2) tensor decompositions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from .actions import GF2ActionSpace
from .game import slice_rank_upper_bound
from .tensor import GF2Tensor


@dataclass(frozen=True)
class BeamNode:
    state: GF2Tensor
    action_indices: tuple[int, ...]
    log_probability: float
    score: float
    predicted_remaining: float


@dataclass(frozen=True)
class SearchTrajectory:
    action_indices: tuple[int, ...]
    residual: GF2Tensor
    score: float
    residual_nonzero: int
    residual_rank_upper_bound: int


@dataclass(frozen=True)
class BeamSearchResult:
    success: bool
    action_indices: tuple[int, ...]
    residual: GF2Tensor
    depth: int
    expanded_nodes: int
    final_beam_size: int
    score: float
    trajectories: tuple[SearchTrajectory, ...] = ()


def _as_trajectory(node: BeamNode) -> SearchTrajectory:
    return SearchTrajectory(
        action_indices=node.action_indices,
        residual=node.state.copy(),
        score=node.score,
        residual_nonzero=int(np.count_nonzero(node.state)),
        residual_rank_upper_bound=slice_rank_upper_bound(node.state),
    )


def _validate_target(target: ArrayLike) -> GF2Tensor:
    values = np.asarray(target)
    if values.shape != (4, 4, 4):
        raise ValueError("target must have shape (4, 4, 4)")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("target may contain only GF(2) values 0 and 1")
    return values.astype(np.uint8, copy=True)


def _find_exact_tail(
    residual: GF2Tensor,
    *,
    used_actions: set[int],
    action_terms: GF2Tensor,
    term_indices: dict[bytes, int],
    max_terms: int,
) -> tuple[int, ...] | None:
    """Find an exact one- or two-action completion of a residual."""
    direct = term_indices.get(residual.tobytes())
    if direct is not None and direct not in used_actions:
        return (direct,)
    if max_terms < 2:
        return None

    for first in range(len(action_terms)):
        if first in used_actions:
            continue
        second = term_indices.get((residual ^ action_terms[first]).tobytes())
        if (
            second is not None
            and second != first
            and second not in used_actions
        ):
            return first, second
    return None


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    states: list[GF2Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        state_array = np.stack(states[start : start + batch_size])
        inputs = torch.from_numpy(state_array).to(device=device, dtype=torch.float32)
        inputs = inputs * 2.0 - 1.0
        logits, values = model(inputs)
        logits_parts.append(logits.cpu())
        value_parts.append(values.cpu())
    return torch.cat(logits_parts), torch.cat(value_parts)


def policy_guided_beam_search(
    model: nn.Module,
    target: ArrayLike,
    *,
    action_space: GF2ActionSpace | None = None,
    max_steps: int = 7,
    beam_width: int = 256,
    top_k: int = 32,
    value_weight: float = 0.1,
    value_candidate_multiplier: int = 4,
    exact_tail_steps: int = 2,
    inference_batch_size: int = 512,
    policy_noise: float = 0.0,
    seed: int = 0,
    replay_trajectory_count: int = 0,
    device: str | torch.device = "cpu",
) -> BeamSearchResult:
    """Search for a short exact decomposition using policy and value guidance.

    At every depth, each beam state expands using its top-k policy actions.
    Equivalent residuals are merged, obvious repeated actions are excluded, and
    the value estimate helps select the next beam.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if value_weight < 0:
        raise ValueError("value_weight must be nonnegative")
    if value_candidate_multiplier < 1 or inference_batch_size < 1:
        raise ValueError("candidate multiplier and inference batch size must be positive")
    if exact_tail_steps not in (0, 1, 2):
        raise ValueError("exact_tail_steps must be 0, 1, or 2")
    if policy_noise < 0:
        raise ValueError("policy_noise must be nonnegative")
    if replay_trajectory_count < 0:
        raise ValueError("replay_trajectory_count must be nonnegative")

    catalog = action_space or GF2ActionSpace()
    if top_k > catalog.size:
        raise ValueError("top_k cannot exceed the action-space size")
    target_state = _validate_target(target)
    if not np.any(target_state):
        empty_node = BeamNode(target_state, (), 0.0, 0.0, 0.0)
        return BeamSearchResult(
            True, (), target_state, 0, 0, 1, 0.0, (_as_trajectory(empty_node),)
        )

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    action_terms = catalog.terms
    term_indices = {
        action_terms[index].tobytes(): index for index in range(catalog.size)
    }
    beam = [BeamNode(target_state, (), 0.0, 0.0, float(max_steps))]
    expanded_nodes = 0
    noise_generator = torch.Generator().manual_seed(seed)

    for depth in range(1, max_steps + 1):
        if exact_tail_steps:
            for node in beam:
                remaining_budget = max_steps - len(node.action_indices)
                if 0 < remaining_budget <= exact_tail_steps:
                    tail = _find_exact_tail(
                        node.state,
                        used_actions=set(node.action_indices),
                        action_terms=action_terms,
                        term_indices=term_indices,
                        max_terms=remaining_budget,
                    )
                    if tail is not None:
                        solved_actions = node.action_indices + tail
                        return BeamSearchResult(
                            True,
                            solved_actions,
                            np.zeros_like(node.state),
                            len(solved_actions),
                            expanded_nodes,
                            len(beam),
                            node.score,
                            (
                                SearchTrajectory(
                                    action_indices=solved_actions,
                                    residual=np.zeros_like(node.state),
                                    score=node.score,
                                    residual_nonzero=0,
                                    residual_rank_upper_bound=0,
                                ),
                            ),
                        )

        logits, _ = _evaluate_model(
            model,
            [node.state for node in beam],
            device=target_device,
            batch_size=inference_batch_size,
        )
        if policy_noise:
            logits = logits + policy_noise * torch.randn(
                logits.shape, generator=noise_generator
            )
        top_log_probabilities, top_indices = torch.log_softmax(logits, dim=1).topk(
            top_k, dim=1
        )

        unique_children: dict[bytes, BeamNode] = {}
        successful: list[BeamNode] = []
        for node_number, node in enumerate(beam):
            used_actions = set(node.action_indices)
            for candidate_number in range(top_k):
                action_index = int(top_indices[node_number, candidate_number])
                if action_index in used_actions:
                    continue
                child_state = node.state ^ action_terms[action_index]
                child_log_probability = node.log_probability + float(
                    top_log_probabilities[node_number, candidate_number]
                )
                child = BeamNode(
                    state=child_state,
                    action_indices=node.action_indices + (action_index,),
                    log_probability=child_log_probability,
                    score=child_log_probability,
                    predicted_remaining=float(max_steps - depth),
                )
                expanded_nodes += 1

                if not np.any(child_state):
                    successful.append(child)
                    continue

                key = child_state.tobytes()
                previous = unique_children.get(key)
                if previous is None or child.log_probability > previous.log_probability:
                    unique_children[key] = child

        if successful:
            winner = max(successful, key=lambda node: node.log_probability)
            return BeamSearchResult(
                True,
                winner.action_indices,
                winner.state,
                depth,
                expanded_nodes,
                len(beam),
                winner.log_probability,
                (_as_trajectory(winner),),
            )
        if not unique_children:
            break

        preliminary_limit = beam_width * value_candidate_multiplier
        candidates = sorted(
            unique_children.values(),
            key=lambda node: node.log_probability,
            reverse=True,
        )[:preliminary_limit]
        _, predicted_values = _evaluate_model(
            model,
            [node.state for node in candidates],
            device=target_device,
            batch_size=inference_batch_size,
        )

        scored_candidates: list[BeamNode] = []
        for node, predicted_value in zip(candidates, predicted_values.tolist()):
            remaining = max(0.0, float(predicted_value))
            scored_candidates.append(
                BeamNode(
                    state=node.state,
                    action_indices=node.action_indices,
                    log_probability=node.log_probability,
                    score=node.log_probability - value_weight * remaining,
                    predicted_remaining=remaining,
                )
            )
        scored_candidates.sort(key=lambda node: node.score, reverse=True)
        beam = scored_candidates[:beam_width]

    best = max(beam, key=lambda node: node.score)
    trajectories: tuple[SearchTrajectory, ...] = ()
    if replay_trajectory_count:
        replay_nodes = sorted(
            beam,
            key=lambda node: (
                slice_rank_upper_bound(node.state),
                int(np.count_nonzero(node.state)),
                -node.score,
            ),
        )[:replay_trajectory_count]
        trajectories = tuple(_as_trajectory(node) for node in replay_nodes)
    return BeamSearchResult(
        False,
        best.action_indices,
        best.state,
        len(best.action_indices),
        expanded_nodes,
        len(beam),
        best.score,
        trajectories,
    )
