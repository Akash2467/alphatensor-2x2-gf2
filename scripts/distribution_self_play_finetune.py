"""Fine-tune with sparse MCTS visit-count policy distributions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.mcts import policy_guided_batched_mcts, policy_guided_mcts
from alphatensor.model import (
    load_policy_value_checkpoint,
    make_data_loader,
    make_mixed_data_loader,
    make_sparse_policy_loader,
    run_epoch,
    run_sparse_policy_epoch,
)
from alphatensor.policy_replay import (
    PolicyTargetReplayBuffer,
    transform_policy_target,
)
from alphatensor.replay import TargetReplayBuffer
from alphatensor.selection import (
    CheckpointQuality,
    is_better_checkpoint,
    outcome_weight,
)
from alphatensor.synthetic import generate_synthetic_dataset
from alphatensor.symmetry import random_basis_transform
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mcts_self_play_transformer.pt"),
    )
    parser.add_argument(
        "--hard-replay", type=Path, default=Path("replay/mcts_replay.npz")
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints/distribution_transformer.pt"),
    )
    parser.add_argument(
        "--policy-replay-output",
        type=Path,
        default=Path("replay/policy_distribution_replay.npz"),
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("results/distribution_history.json"),
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--batches-per-cycle", type=int, default=2)
    parser.add_argument("--num-trees", type=int, default=4)
    parser.add_argument("--simulations-per-tree", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--policy-targets-per-tree", type=int, default=256)
    parser.add_argument("--policy-actions", type=int, default=16)
    parser.add_argument("--symmetry-augmentations", type=int, default=2)
    parser.add_argument("--policy-replay-capacity", type=int, default=50_000)
    parser.add_argument("--hard-replay-fraction", type=float, default=0.25)
    parser.add_argument("--synthetic-games", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--evaluation-simulations", type=int, default=5_000)
    parser.add_argument("--max-accepted-residual-rank", type=int, default=2)
    parser.add_argument("--quality-temperature", type=float, default=0.75)
    parser.add_argument(
        "--rollback-on-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: dict[str, object],
    cycle: int,
    hard_replay_size: int,
    policy_replay_size: int,
    quality: CheckpointQuality,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": source["architecture"],
            "model_config": source["model_config"],
            "source_mcts_self_play_cycle": source.get("mcts_self_play_cycle"),
            "distribution_self_play_cycle": cycle,
            "replay_size": hard_replay_size,
            "policy_replay_size": policy_replay_size,
            "selected_quality": {
                "search_success": quality.search_success,
                "residual_rank_upper_bound": quality.residual_rank_upper_bound,
                "residual_nonzero": quality.residual_nonzero,
                "validation_top5": quality.validation_top5,
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.cycles,
        args.batches_per_cycle,
        args.num_trees,
        args.simulations_per_tree,
        args.epochs,
    ) < 1:
        raise ValueError("cycle, tree, simulation, and epoch counts must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, source = load_policy_value_checkpoint(args.checkpoint, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    hard_replay = TargetReplayBuffer.load(args.hard_replay)
    policy_replay = PolicyTargetReplayBuffer(
        capacity=args.policy_replay_capacity,
        max_actions_per_state=args.policy_actions,
    )
    validation = generate_synthetic_dataset(1_000, ranks=[5, 6, 7], seed=43)
    validation_loader = make_data_loader(
        validation, batch_size=args.batch_size, shuffle=False, seed=43
    )
    history: list[dict[str, object]] = []

    baseline_validation = run_epoch(model, validation_loader, device=device)
    baseline_evaluation = policy_guided_mcts(
        model,
        target,
        action_space=action_space,
        simulations=args.evaluation_simulations,
        top_k=args.top_k,
        root_noise_fraction=0.0,
        exact_tail_steps=2,
        seed=args.seed,
        device=device,
    )
    best_quality = CheckpointQuality(
        baseline_evaluation.success,
        baseline_evaluation.residual_rank_upper_bound,
        baseline_evaluation.residual_nonzero,
        baseline_validation.policy_top5_accuracy,
    )
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_cycle = 0

    print(
        f"device={device} cycles={args.cycles} hard_replay={len(hard_replay)} "
        f"baseline_top5={best_quality.validation_top5:.3%}"
    )
    for cycle in range(1, args.cycles + 1):
        new_states = 0
        search_summaries: list[dict[str, object]] = []
        for batch in range(args.batches_per_cycle):
            results = policy_guided_batched_mcts(
                model,
                target,
                num_trees=args.num_trees,
                simulations_per_tree=args.simulations_per_tree,
                action_space=action_space,
                top_k=args.top_k,
                root_noise_fraction=0.45,
                replay_trajectory_count=0,
                policy_target_count=args.policy_targets_per_tree,
                policy_target_top_actions=args.policy_actions,
                seed=args.seed + cycle * 10_000 + batch * 100,
                device=device,
            )
            for result in results:
                accepted = (
                    result.success
                    or result.residual_rank_upper_bound
                    <= args.max_accepted_residual_rank
                )
                weight = outcome_weight(
                    success=result.success,
                    residual_rank_upper_bound=result.residual_rank_upper_bound,
                    residual_nonzero=result.residual_nonzero,
                    temperature=args.quality_temperature,
                )
                if accepted:
                    for policy_target in result.policy_targets:
                        new_states += int(policy_replay.add(policy_target, weight=weight))
                    for _ in range(args.symmetry_augmentations):
                        transform = random_basis_transform(rng)
                        for policy_target in result.policy_targets:
                            new_states += int(
                                policy_replay.add(
                                    transform_policy_target(
                                        policy_target,
                                        transform,
                                        action_space=action_space,
                                    ),
                                    weight=weight,
                                )
                            )
                search_summaries.append(
                    {
                        "success": result.success,
                        "expanded_nodes": result.expanded_nodes,
                        "policy_targets": len(result.policy_targets),
                        "residual_nonzero": result.residual_nonzero,
                        "residual_rank_upper_bound": result.residual_rank_upper_bound,
                        "accepted": accepted,
                        "outcome_weight": weight,
                    }
                )
            print(
                f"cycle={cycle} batch={batch + 1}/{args.batches_per_cycle} "
                f"trees={len(results)} policy_replay={len(policy_replay)}"
            )
            if any(result.success for result in results):
                break

        args.policy_replay_output.parent.mkdir(parents=True, exist_ok=True)
        policy_replay.save(args.policy_replay_output)
        synthetic = generate_synthetic_dataset(
            args.synthetic_games, ranks=[5, 6, 7], seed=args.seed + cycle
        )
        hard_loader = make_mixed_data_loader(
            synthetic,
            hard_replay.arrays(),
            replay_fraction=args.hard_replay_fraction,
            batch_size=args.batch_size,
            seed=args.seed + cycle,
        )
        soft_loader = make_sparse_policy_loader(
            policy_replay.arrays(),
            batch_size=args.batch_size,
            seed=args.seed + cycle,
        )
        epoch_summaries: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            hard_metrics = run_epoch(
                model, hard_loader, optimizer=optimizer, device=device
            )
            soft_metrics = run_sparse_policy_epoch(
                model, soft_loader, optimizer=optimizer, device=device
            )
            epoch_summaries.append(
                {
                    "hard_loss": hard_metrics.loss,
                    "soft_loss": soft_metrics.loss,
                    "soft_top5_any": soft_metrics.policy_top5_accuracy,
                }
            )
            print(
                f"cycle={cycle} epoch={epoch} hard_loss={hard_metrics.loss:.4f} "
                f"soft_loss={soft_metrics.loss:.4f} "
                f"soft_top5_any={soft_metrics.policy_top5_accuracy:.3%}"
            )

        validation_metrics = run_epoch(model, validation_loader, device=device)
        evaluation = policy_guided_mcts(
            model,
            target,
            action_space=action_space,
            simulations=args.evaluation_simulations,
            top_k=args.top_k,
            root_noise_fraction=0.0,
            exact_tail_steps=2,
            seed=args.seed,
            device=device,
        )
        candidate_quality = CheckpointQuality(
            evaluation.success,
            evaluation.residual_rank_upper_bound,
            evaluation.residual_nonzero,
            validation_metrics.policy_top5_accuracy,
        )
        selected = is_better_checkpoint(candidate_quality, best_quality)
        if selected:
            best_quality = candidate_quality
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_cycle = cycle
        summary = {
            "cycle": cycle,
            "new_policy_states": new_states,
            "policy_replay_size": len(policy_replay),
            "validation_top1": validation_metrics.policy_accuracy,
            "validation_top5": validation_metrics.policy_top5_accuracy,
            "searches": search_summaries,
            "epochs": epoch_summaries,
            "evaluation": {
                "success": evaluation.success,
                "action_indices": list(evaluation.action_indices),
                "expanded_nodes": evaluation.expanded_nodes,
                "residual_nonzero": evaluation.residual_nonzero,
                "residual_rank_upper_bound": evaluation.residual_rank_upper_bound,
            },
            "checkpoint_selected": selected,
            "best_cycle_after_evaluation": best_cycle,
        }
        history.append(summary)
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"cycle={cycle} validation_top5={validation_metrics.policy_top5_accuracy:.3%} "
            f"evaluation_success={evaluation.success} selected={selected}"
        )
        if evaluation.success:
            break
        if not selected and args.rollback_on_regression:
            model.load_state_dict(best_model_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            optimizer.load_state_dict(best_optimizer_state)

    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    save_checkpoint(
        args.output_checkpoint,
        model,
        optimizer,
        source,
        best_cycle,
        len(hard_replay),
        len(policy_replay),
        best_quality,
    )

    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"policy_replay={args.policy_replay_output.resolve()}")
    print(f"history={args.history_output.resolve()}")


if __name__ == "__main__":
    main()
