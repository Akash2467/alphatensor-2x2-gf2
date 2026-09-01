"""Policy-guided Monte Carlo tree search for the 2 x 2 TensorGame."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from .actions import GF2ActionSpace
from .game import slice_rank_upper_bound
from .search import SearchTrajectory
from .tensor import GF2Tensor


@dataclass
class _TreeNode:
    state: GF2Tensor
    action_indices: tuple[int, ...]
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    candidates: tuple[tuple[int, float], ...] | None = None
    children: dict[int, "_TreeNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class MCTSPolicyTarget:
    state: GF2Tensor
    action_indices: tuple[int, ...]
    visit_counts: tuple[int, ...]
    remaining_steps: float


@dataclass(frozen=True)
class MCTSSearchResult:
    success: bool
    action_indices: tuple[int, ...]
    residual: GF2Tensor
    simulations: int
    expanded_nodes: int
    root_visits: int
    score: float
    residual_nonzero: int
    residual_rank_upper_bound: int
    trajectories: tuple[SearchTrajectory, ...] = ()
    policy_targets: tuple[MCTSPolicyTarget, ...] = ()


def _as_trajectory(node: _TreeNode) -> SearchTrajectory:
    return SearchTrajectory(
        action_indices=node.action_indices,
        residual=node.state.copy(),
        score=node.mean_value + np.log1p(node.visits),
        residual_nonzero=int(np.count_nonzero(node.state)),
        residual_rank_upper_bound=slice_rank_upper_bound(node.state),
    )


def _collect_policy_targets(
    nodes: list[_TreeNode],
    *,
    max_steps: int,
    target_count: int,
    top_actions: int,
) -> tuple[MCTSPolicyTarget, ...]:
    eligible = [
        node
        for node in nodes
        if node.children and any(child.visits for child in node.children.values())
    ]
    eligible.sort(key=lambda node: node.visits, reverse=True)
    targets: list[MCTSPolicyTarget] = []
    for node in eligible[:target_count]:
        visited = sorted(
            (
                (action, child.visits)
                for action, child in node.children.items()
                if child.visits > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_actions]
        if not visited:
            continue
        targets.append(
            MCTSPolicyTarget(
                state=node.state.copy(),
                action_indices=tuple(action for action, _ in visited),
                visit_counts=tuple(visits for _, visits in visited),
                remaining_steps=float(max_steps - len(node.action_indices)),
            )
        )
    return tuple(targets)


def _validate_target(target: ArrayLike) -> GF2Tensor:
    values = np.asarray(target)
    if values.shape != (4, 4, 4):
        raise ValueError("target must have shape (4, 4, 4)")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("target must contain only GF(2) values")
    return values.astype(np.uint8, copy=True)


def _find_exact_tail(
    residual: GF2Tensor,
    *,
    used_actions: set[int],
    action_terms: GF2Tensor,
    term_indices: dict[bytes, int],
    max_terms: int,
) -> tuple[int, ...] | None:
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
def _expand_nodes(
    nodes: list[_TreeNode],
    model: nn.Module,
    *,
    top_k: int,
    action_count: int,
    value_blend: float,
    root_noise_fraction: float,
    dirichlet_alpha: float,
    rngs: list[np.random.Generator],
    device: torch.device,
) -> list[float]:
    if len(nodes) != len(rngs) or not nodes:
        raise ValueError("nodes and rngs must be nonempty and aligned")
    state_batch = np.stack([node.state for node in nodes])
    inputs = torch.from_numpy(state_batch).to(device=device, dtype=torch.float32)
    logits_batch, predicted_batch = model(inputs.mul(2.0).sub(1.0))
    values: list[float] = []
    for node, logits, predicted_remaining, rng in zip(
        nodes, logits_batch.detach().cpu(), predicted_batch.detach().cpu(), rngs
    ):
        used = set(node.action_indices)
        if used:
            logits[list(used)] = -torch.inf
        count = min(top_k, action_count - len(used))
        top_values, top_indices = logits.topk(count)
        priors = torch.softmax(top_values, dim=0).numpy()
        if not node.action_indices and root_noise_fraction:
            noise = rng.dirichlet(np.full(count, dirichlet_alpha))
            priors = (1.0 - root_noise_fraction) * priors + root_noise_fraction * noise
        priors /= priors.sum()
        node.candidates = tuple(
            (int(action), float(prior))
            for action, prior in zip(top_indices.tolist(), priors.tolist())
        )
        predicted = max(0.0, float(predicted_remaining))
        predicted_score = 1.0 - min(predicted / 8.0, 1.0)
        rank_score = 1.0 - min(slice_rank_upper_bound(node.state) / 8.0, 1.0)
        values.append(
            value_blend * predicted_score + (1.0 - value_blend) * rank_score
        )
    return values


def _expand_node(
    node: _TreeNode,
    model: nn.Module,
    *,
    top_k: int,
    action_count: int,
    value_blend: float,
    root_noise_fraction: float,
    dirichlet_alpha: float,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    return _expand_nodes(
        [node], model, top_k=top_k, action_count=action_count,
        value_blend=value_blend, root_noise_fraction=root_noise_fraction,
        dirichlet_alpha=dirichlet_alpha, rngs=[rng], device=device,
    )[0]


def _select_action(node: _TreeNode, c_puct: float) -> tuple[int, float]:
    if not node.candidates:
        raise ValueError("cannot select from an unexpanded node")
    exploration_scale = sqrt(node.visits + 1)
    best_action = -1
    best_prior = 0.0
    best_score = -float("inf")
    for action, prior in node.candidates:
        child = node.children.get(action)
        visits = child.visits if child is not None else 0
        mean_value = child.mean_value if child is not None else 0.0
        score = mean_value + c_puct * prior * exploration_scale / (1 + visits)
        if score > best_score:
            best_action, best_prior, best_score = action, prior, score
    return best_action, best_prior


def policy_guided_mcts(
    model: nn.Module,
    target: ArrayLike,
    *,
    action_space: GF2ActionSpace | None = None,
    max_steps: int = 7,
    simulations: int = 5_000,
    top_k: int = 128,
    c_puct: float = 1.5,
    value_blend: float = 0.5,
    root_noise_fraction: float = 0.25,
    dirichlet_alpha: float = 0.3,
    exact_tail_steps: int = 2,
    replay_trajectory_count: int = 0,
    policy_target_count: int = 0,
    policy_target_top_actions: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> MCTSSearchResult:
    """Search for an exact decomposition using PUCT-guided tree exploration."""
    if max_steps < 1 or simulations < 1:
        raise ValueError("max_steps and simulations must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if c_puct <= 0:
        raise ValueError("c_puct must be positive")
    if not 0.0 <= value_blend <= 1.0:
        raise ValueError("value_blend must be between 0 and 1")
    if not 0.0 <= root_noise_fraction <= 1.0:
        raise ValueError("root_noise_fraction must be between 0 and 1")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    if exact_tail_steps not in (0, 1, 2):
        raise ValueError("exact_tail_steps must be 0, 1, or 2")
    if replay_trajectory_count < 0:
        raise ValueError("replay_trajectory_count must be nonnegative")
    if policy_target_count < 0 or policy_target_top_actions < 1:
        raise ValueError("policy target count must be nonnegative and top actions positive")

    catalog = action_space or GF2ActionSpace()
    if top_k > catalog.size:
        raise ValueError("top_k cannot exceed the action-space size")
    target_state = _validate_target(target)
    if not np.any(target_state):
        return MCTSSearchResult(True, (), target_state, 0, 0, 0, 2.0, 0, 0)

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    rng = np.random.default_rng(seed)
    action_terms = catalog.terms
    term_indices = {
        action_terms[index].tobytes(): index for index in range(catalog.size)
    }
    root = _TreeNode(target_state, ())
    all_nodes = [root]
    expanded_nodes = 0
    completed_simulations = 0
    best = root

    def best_key(node: _TreeNode) -> tuple[int, int, int]:
        return (
            len(node.action_indices) + slice_rank_upper_bound(node.state),
            int(np.count_nonzero(node.state)),
            -len(node.action_indices),
        )

    for simulation in range(1, simulations + 1):
        node = root
        path = [root]
        leaf_value = 0.0

        while True:
            depth = len(node.action_indices)
            if not np.any(node.state):
                leaf_value = 2.0
                break
            if depth >= max_steps:
                leaf_value = 1.0 - min(
                    slice_rank_upper_bound(node.state) / 8.0, 1.0
                )
                break
            if node.candidates is None:
                leaf_value = _expand_node(
                    node,
                    model,
                    top_k=top_k,
                    action_count=catalog.size,
                    value_blend=value_blend,
                    root_noise_fraction=root_noise_fraction,
                    dirichlet_alpha=dirichlet_alpha,
                    rng=rng,
                    device=target_device,
                )
                expanded_nodes += 1
                break

            action, prior = _select_action(node, c_puct)
            child = node.children.get(action)
            if child is None:
                child = _TreeNode(
                    state=node.state ^ action_terms[action],
                    action_indices=node.action_indices + (action,),
                    prior=prior,
                )
                node.children[action] = child
                all_nodes.append(child)
            node = child
            path.append(node)
            if best_key(node) < best_key(best):
                best = node

            remaining = max_steps - len(node.action_indices)
            if exact_tail_steps and 0 < remaining <= exact_tail_steps:
                tail = _find_exact_tail(
                    node.state,
                    used_actions=set(node.action_indices),
                    action_terms=action_terms,
                    term_indices=term_indices,
                    max_terms=remaining,
                )
                if tail is not None:
                    solved = node.action_indices + tail
                    for visited in path:
                        visited.visits += 1
                        visited.value_sum += 2.0
                    policy_targets = _collect_policy_targets(
                        all_nodes,
                        max_steps=max_steps,
                        target_count=policy_target_count,
                        top_actions=policy_target_top_actions,
                    )
                    return MCTSSearchResult(
                        True,
                        solved,
                        np.zeros_like(target_state),
                        simulation,
                        expanded_nodes,
                        root.visits,
                        2.0,
                        0,
                        0,
                        (
                            SearchTrajectory(
                                action_indices=solved,
                                residual=np.zeros_like(target_state),
                                score=2.0,
                                residual_nonzero=0,
                                residual_rank_upper_bound=0,
                            ),
                        ),
                        policy_targets,
                    )

        for visited in path:
            visited.visits += 1
            visited.value_sum += leaf_value
        completed_simulations = simulation

    trajectories: tuple[SearchTrajectory, ...] = ()
    if replay_trajectory_count:
        terminal_nodes = [
            node for node in all_nodes if len(node.action_indices) == max_steps
        ]
        terminal_nodes.sort(
            key=lambda node: (
                slice_rank_upper_bound(node.state),
                int(np.count_nonzero(node.state)),
                -node.visits,
                -node.mean_value,
            )
        )
        trajectories = tuple(
            _as_trajectory(node)
            for node in terminal_nodes[:replay_trajectory_count]
        )
    policy_targets = _collect_policy_targets(
        all_nodes,
        max_steps=max_steps,
        target_count=policy_target_count,
        top_actions=policy_target_top_actions,
    )
    return MCTSSearchResult(
        False,
        best.action_indices,
        best.state.copy(),
        completed_simulations,
        expanded_nodes,
        root.visits,
        best.mean_value,
        int(np.count_nonzero(best.state)),
        slice_rank_upper_bound(best.state),
        trajectories,
        policy_targets,
    )


def policy_guided_batched_mcts(
    model: nn.Module,
    target: ArrayLike,
    *,
    num_trees: int = 4,
    simulations_per_tree: int = 1_000,
    action_space: GF2ActionSpace | None = None,
    max_steps: int = 7,
    top_k: int = 128,
    c_puct: float = 2.0,
    value_blend: float = 0.5,
    root_noise_fraction: float = 0.4,
    dirichlet_alpha: float = 0.3,
    exact_tail_steps: int = 2,
    replay_trajectory_count: int = 64,
    policy_target_count: int = 256,
    policy_target_top_actions: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[MCTSSearchResult, ...]:
    """Run independent MCTS trees while batching one leaf evaluation per tree."""
    if num_trees < 1 or simulations_per_tree < 1:
        raise ValueError("num_trees and simulations_per_tree must be positive")
    if not 1 <= top_k <= (action_space.size if action_space else GF2ActionSpace.size):
        raise ValueError("top_k is outside the action-space range")
    catalog = action_space or GF2ActionSpace()
    target_state = _validate_target(target)
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    action_terms = catalog.terms
    term_indices = {
        action_terms[index].tobytes(): index for index in range(catalog.size)
    }

    contexts: list[dict[str, object]] = []
    for tree in range(num_trees):
        root = _TreeNode(target_state.copy(), ())
        contexts.append(
            {
                "root": root,
                "nodes": [root],
                "best": root,
                "expanded": 0,
                "rng": np.random.default_rng(seed + tree * 104729),
            }
        )

    def node_key(node: _TreeNode) -> tuple[int, int, int]:
        return (
            len(node.action_indices) + slice_rank_upper_bound(node.state),
            int(np.count_nonzero(node.state)),
            -len(node.action_indices),
        )

    for simulation in range(1, simulations_per_tree + 1):
        pending: list[tuple[dict[str, object], _TreeNode, list[_TreeNode]]] = []
        resolved: list[tuple[list[_TreeNode], float]] = []
        for context in contexts:
            node = context["root"]
            assert isinstance(node, _TreeNode)
            path = [node]
            while True:
                depth = len(node.action_indices)
                if not np.any(node.state):
                    resolved.append((path, 2.0))
                    break
                if depth >= max_steps:
                    value = 1.0 - min(slice_rank_upper_bound(node.state) / 8.0, 1.0)
                    resolved.append((path, value))
                    break
                if node.candidates is None:
                    pending.append((context, node, path))
                    break
                action, prior = _select_action(node, c_puct)
                child = node.children.get(action)
                if child is None:
                    child = _TreeNode(
                        node.state ^ action_terms[action],
                        node.action_indices + (action,),
                        prior=prior,
                    )
                    node.children[action] = child
                    nodes = context["nodes"]
                    assert isinstance(nodes, list)
                    nodes.append(child)
                node = child
                path.append(node)
                best = context["best"]
                assert isinstance(best, _TreeNode)
                if node_key(node) < node_key(best):
                    context["best"] = node
                remaining = max_steps - len(node.action_indices)
                if exact_tail_steps and 0 < remaining <= exact_tail_steps:
                    tail = _find_exact_tail(
                        node.state,
                        used_actions=set(node.action_indices),
                        action_terms=action_terms,
                        term_indices=term_indices,
                        max_terms=remaining,
                    )
                    if tail is not None:
                        solved = node.action_indices + tail
                        for visited in path:
                            visited.visits += 1
                            visited.value_sum += 2.0
                        nodes = context["nodes"]
                        assert isinstance(nodes, list)
                        policy_targets = _collect_policy_targets(
                            nodes,
                            max_steps=max_steps,
                            target_count=policy_target_count,
                            top_actions=policy_target_top_actions,
                        )
                        trajectory = SearchTrajectory(
                            solved, np.zeros_like(target_state), 2.0, 0, 0
                        )
                        return (
                            MCTSSearchResult(
                                True, solved, np.zeros_like(target_state), simulation,
                                int(context["expanded"]), path[0].visits, 2.0, 0, 0,
                                (trajectory,), policy_targets,
                            ),
                        )

        if pending:
            pending_nodes = [item[1] for item in pending]
            pending_rngs = [item[0]["rng"] for item in pending]
            assert all(isinstance(rng, np.random.Generator) for rng in pending_rngs)
            leaf_values = _expand_nodes(
                pending_nodes,
                model,
                top_k=top_k,
                action_count=catalog.size,
                value_blend=value_blend,
                root_noise_fraction=root_noise_fraction,
                dirichlet_alpha=dirichlet_alpha,
                rngs=pending_rngs,  # type: ignore[arg-type]
                device=target_device,
            )
            for (context, _, path), value in zip(pending, leaf_values):
                context["expanded"] = int(context["expanded"]) + 1
                resolved.append((path, value))
        for path, value in resolved:
            for visited in path:
                visited.visits += 1
                visited.value_sum += value

    results: list[MCTSSearchResult] = []
    for context in contexts:
        root = context["root"]
        best = context["best"]
        nodes = context["nodes"]
        assert isinstance(root, _TreeNode) and isinstance(best, _TreeNode)
        assert isinstance(nodes, list)
        terminal_nodes = [node for node in nodes if len(node.action_indices) == max_steps]
        terminal_nodes.sort(
            key=lambda node: (
                slice_rank_upper_bound(node.state),
                int(np.count_nonzero(node.state)),
                -node.visits,
            )
        )
        trajectories = tuple(
            _as_trajectory(node)
            for node in terminal_nodes[:replay_trajectory_count]
        )
        policy_targets = _collect_policy_targets(
            nodes,
            max_steps=max_steps,
            target_count=policy_target_count,
            top_actions=policy_target_top_actions,
        )
        results.append(
            MCTSSearchResult(
                False,
                best.action_indices,
                best.state.copy(),
                simulations_per_tree,
                int(context["expanded"]),
                root.visits,
                best.mean_value,
                int(np.count_nonzero(best.state)),
                slice_rank_upper_bound(best.state),
                trajectories,
                policy_targets,
            )
        )
    return tuple(results)
