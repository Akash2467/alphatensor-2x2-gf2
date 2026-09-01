"""Close the MCTS self-play loop with replay collection and fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.mcts import MCTSSearchResult, policy_guided_mcts
from alphatensor.model import (
    load_policy_value_checkpoint,
    make_data_loader,
    make_mixed_data_loader,
    run_epoch,
)
from alphatensor.replay import TargetReplayBuffer
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
        "--checkpoint", type=Path, default=Path("checkpoints/symmetry_transformer.pt")
    )
    parser.add_argument(
        "--initial-replay", type=Path, default=Path("replay/symmetry_replay.npz")
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints/mcts_self_play_transformer.pt"),
    )
    parser.add_argument(
        "--replay-output", type=Path, default=Path("replay/mcts_replay.npz")
    )
    parser.add_argument(
        "--history-output", type=Path, default=Path("results/mcts_self_play_history.json")
    )
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--mcts-runs", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=3_000)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--root-noise-fraction", type=float, default=0.4)
    parser.add_argument("--trajectories-per-run", type=int, default=64)
    parser.add_argument("--symmetry-augmentations", type=int, default=2)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--replay-fraction", type=float, default=0.4)
    parser.add_argument("--synthetic-games", type=int, default=5_000)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--evaluation-simulations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def result_summary(result: MCTSSearchResult) -> dict[str, object]:
    return {
        "success": result.success,
        "action_indices": list(result.action_indices),
        "simulations": result.simulations,
        "expanded_nodes": result.expanded_nodes,
        "residual_nonzero": result.residual_nonzero,
        "residual_rank_upper_bound": result.residual_rank_upper_bound,
        "retained_trajectories": len(result.trajectories),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: dict[str, object],
    cycle: int,
    replay_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": source["architecture"],
            "model_config": source["model_config"],
            "source_checkpoint_epoch": source.get("epoch"),
            "source_self_play_cycle": source.get("self_play_cycle"),
            "mcts_self_play_cycle": cycle,
            "replay_size": replay_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if min(args.cycles, args.mcts_runs, args.simulations, args.finetune_epochs) < 1:
        raise ValueError("cycles, runs, simulations, and epochs must be positive")
    if args.symmetry_augmentations < 0:
        raise ValueError("symmetry-augmentations must be nonnegative")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, source = load_policy_value_checkpoint(args.checkpoint, device=device)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    replay = TargetReplayBuffer.load(args.initial_replay)
    replay.capacity = args.replay_capacity
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    validation = generate_synthetic_dataset(1_000, ranks=[5, 6, 7], seed=43)
    validation_loader = make_data_loader(
        validation, batch_size=args.batch_size, shuffle=False, seed=43
    )
    history: list[dict[str, object]] = []

    print(f"device={device} cycles={args.cycles} initial_replay={len(replay)}")
    for cycle in range(1, args.cycles + 1):
        changed = 0
        searches: list[dict[str, object]] = []
        for run in range(args.mcts_runs):
            result = policy_guided_mcts(
                model,
                target,
                action_space=action_space,
                simulations=args.simulations,
                top_k=args.top_k,
                c_puct=args.c_puct,
                root_noise_fraction=args.root_noise_fraction,
                exact_tail_steps=2,
                replay_trajectory_count=args.trajectories_per_run,
                seed=args.seed + cycle * 1_000 + run,
                device=device,
            )
            for trajectory in result.trajectories:
                changed += replay.add_trajectory(
                    target, trajectory, action_space=action_space
                )
                for _ in range(args.symmetry_augmentations):
                    transform = random_basis_transform(rng)
                    changed += replay.add_trajectory(
                        transform_tensor(target, transform),
                        transform_trajectory(
                            trajectory, transform, action_space=action_space
                        ),
                        action_space=action_space,
                    )
            searches.append(result_summary(result))
            print(
                f"cycle={cycle} mcts={run + 1}/{args.mcts_runs} "
                f"success={result.success} trajectories={len(result.trajectories)} "
                f"replay={len(replay)}"
            )
            if result.success:
                break

        if not replay.arrays().action_indices.size:
            raise RuntimeError("MCTS produced no usable replay trajectories")
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        replay.save(args.replay_output)
        synthetic = generate_synthetic_dataset(
            args.synthetic_games, ranks=[5, 6, 7], seed=args.seed + cycle
        )
        loader = make_mixed_data_loader(
            synthetic,
            replay.arrays(),
            replay_fraction=args.replay_fraction,
            batch_size=args.batch_size,
            seed=args.seed + cycle,
        )
        for epoch in range(1, args.finetune_epochs + 1):
            metrics = run_epoch(model, loader, optimizer=optimizer, device=device)
            print(
                f"cycle={cycle} epoch={epoch} loss={metrics.loss:.4f} "
                f"top5={metrics.policy_top5_accuracy:.3%}"
            )

        validation_metrics = run_epoch(model, validation_loader, device=device)
        evaluation = policy_guided_mcts(
            model,
            target,
            action_space=action_space,
            simulations=args.evaluation_simulations,
            top_k=args.top_k,
            c_puct=args.c_puct,
            root_noise_fraction=0.0,
            exact_tail_steps=2,
            seed=args.seed,
            device=device,
        )
        save_checkpoint(
            args.output_checkpoint,
            model=model,
            optimizer=optimizer,
            source=source,
            cycle=cycle,
            replay_size=len(replay),
        )
        summary = {
            "cycle": cycle,
            "new_or_improved_replay_records": changed,
            "replay_size": len(replay),
            "synthetic_validation_top1": validation_metrics.policy_accuracy,
            "synthetic_validation_top5": validation_metrics.policy_top5_accuracy,
            "searches": searches,
            "evaluation": result_summary(evaluation),
        }
        history.append(summary)
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"cycle={cycle} validation_top5={validation_metrics.policy_top5_accuracy:.3%} "
            f"evaluation_success={evaluation.success}"
        )
        if evaluation.success:
            break

    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"replay={args.replay_output.resolve()}")
    print(f"history={args.history_output.resolve()}")


if __name__ == "__main__":
    main()
