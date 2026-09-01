"""Collect target replay trajectories and fine-tune the transformer in cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.model import (
    load_policy_value_checkpoint,
    make_data_loader,
    make_mixed_data_loader,
    run_epoch,
)
from alphatensor.replay import TargetReplayBuffer
from alphatensor.search import BeamSearchResult, policy_guided_beam_search
from alphatensor.synthetic import generate_synthetic_dataset
from alphatensor.symmetry import (
    random_basis_transform,
    transform_tensor,
    transform_trajectory,
)
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/transformer_policy_value.pt"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints/self_play_transformer.pt"),
    )
    parser.add_argument(
        "--replay-output", type=Path, default=Path("replay/target_replay.npz")
    )
    parser.add_argument(
        "--history-output", type=Path, default=Path("results/self_play_history.json")
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--search-runs", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--policy-noise", type=float, default=0.25)
    parser.add_argument("--trajectories-per-search", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--max-actions-per-state", type=int, default=8)
    parser.add_argument("--symmetry-augmentations", type=int, default=4)
    parser.add_argument("--initial-replay", type=Path)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--synthetic-games", type=int, default=5_000)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--evaluation-beam-width", type=int, default=1_024)
    parser.add_argument("--evaluation-top-k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def result_summary(result: BeamSearchResult) -> dict[str, object]:
    return {
        "success": result.success,
        "depth": result.depth,
        "expanded_nodes": result.expanded_nodes,
        "residual_nonzero": int(torch.from_numpy(result.residual).count_nonzero()),
        "action_indices": list(result.action_indices),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: dict[str, object],
    cycle: int,
    replay_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": source_checkpoint["architecture"],
            "model_config": source_checkpoint["model_config"],
            "source_checkpoint_epoch": source_checkpoint.get("epoch"),
            "self_play_cycle": cycle,
            "replay_size": replay_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.cycles < 1 or args.search_runs < 1 or args.finetune_epochs < 1:
        raise ValueError("cycles, search-runs, and finetune-epochs must be positive")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, source_checkpoint = load_policy_value_checkpoint(
        args.checkpoint, device=device
    )
    if source_checkpoint.get("architecture") != "transformer":
        raise ValueError("self-play fine-tuning requires a transformer checkpoint")

    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    if args.symmetry_augmentations < 0:
        raise ValueError("symmetry-augmentations must be nonnegative")
    if args.initial_replay is not None:
        replay = TargetReplayBuffer.load(args.initial_replay)
        replay.capacity = args.replay_capacity
        replay.max_actions_per_state = args.max_actions_per_state
    else:
        replay = TargetReplayBuffer(
            capacity=args.replay_capacity,
            max_actions_per_state=args.max_actions_per_state,
        )
    numpy_rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    validation = generate_synthetic_dataset(1_000, ranks=[5, 6, 7], seed=43)
    validation_loader = make_data_loader(
        validation, batch_size=args.batch_size, shuffle=False, seed=43
    )
    history: list[dict[str, object]] = []

    print(f"device={device} cycles={args.cycles}")
    for cycle in range(1, args.cycles + 1):
        collected = 0
        search_summaries: list[dict[str, object]] = []
        for run in range(args.search_runs):
            result = policy_guided_beam_search(
                model,
                target,
                action_space=action_space,
                max_steps=7,
                beam_width=args.beam_width,
                top_k=args.top_k,
                value_weight=0.0,
                exact_tail_steps=2,
                policy_noise=args.policy_noise,
                seed=args.seed + cycle * 1_000 + run,
                replay_trajectory_count=args.trajectories_per_search,
                device=device,
            )
            for trajectory in result.trajectories:
                collected += replay.add_trajectory(
                    target, trajectory, action_space=action_space
                )
                for _ in range(args.symmetry_augmentations):
                    transform = random_basis_transform(numpy_rng)
                    augmented_target = transform_tensor(target, transform)
                    augmented_trajectory = transform_trajectory(
                        trajectory, transform, action_space=action_space
                    )
                    collected += replay.add_trajectory(
                        augmented_target,
                        augmented_trajectory,
                        action_space=action_space,
                    )
            search_summaries.append(result_summary(result))
            print(
                f"cycle={cycle} search={run + 1}/{args.search_runs} "
                f"success={result.success} replay={len(replay)}"
            )
            if result.success:
                break

        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        replay.save(args.replay_output)

        synthetic = generate_synthetic_dataset(
            args.synthetic_games,
            ranks=[5, 6, 7],
            seed=args.seed + cycle,
        )
        mixed_loader = make_mixed_data_loader(
            synthetic,
            replay.arrays(),
            replay_fraction=args.replay_fraction,
            batch_size=args.batch_size,
            seed=args.seed + cycle,
        )
        training_metrics = None
        for epoch in range(1, args.finetune_epochs + 1):
            training_metrics = run_epoch(
                model, mixed_loader, optimizer=optimizer, device=device
            )
            print(
                f"cycle={cycle} finetune_epoch={epoch} "
                f"loss={training_metrics.loss:.4f} "
                f"top5={training_metrics.policy_top5_accuracy:.3%}"
            )

        validation_metrics = run_epoch(model, validation_loader, device=device)
        evaluation = policy_guided_beam_search(
            model,
            target,
            action_space=action_space,
            max_steps=7,
            beam_width=args.evaluation_beam_width,
            top_k=args.evaluation_top_k,
            value_weight=0.0,
            exact_tail_steps=2,
            device=device,
        )
        save_checkpoint(
            args.output_checkpoint,
            model=model,
            optimizer=optimizer,
            source_checkpoint=source_checkpoint,
            cycle=cycle,
            replay_size=len(replay),
        )
        cycle_summary = {
            "cycle": cycle,
            "new_or_improved_replay_records": collected,
            "replay_size": len(replay),
            "symmetry_augmentations": args.symmetry_augmentations,
            "max_actions_per_state": args.max_actions_per_state,
            "synthetic_validation_top1": validation_metrics.policy_accuracy,
            "synthetic_validation_top5": validation_metrics.policy_top5_accuracy,
            "searches": search_summaries,
            "evaluation": result_summary(evaluation),
        }
        history.append(cycle_summary)
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"cycle={cycle} validation_top5="
            f"{validation_metrics.policy_top5_accuracy:.3%} "
            f"evaluation_success={evaluation.success}"
        )
        if evaluation.success:
            print("rank-7 decomposition discovered")
            break

    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"replay={args.replay_output.resolve()}")
    print(f"history={args.history_output.resolve()}")


if __name__ == "__main__":
    main()
