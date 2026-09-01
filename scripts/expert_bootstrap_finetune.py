"""Fine-tune the transformer on solver-discovered rank-7 decompositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.algorithm import exhaustive_verify_2x2
from alphatensor.decomposition import verify_decomposition
from alphatensor.expert import generate_expert_policy_targets, load_exact_solutions
from alphatensor.game import slice_rank_upper_bound
from alphatensor.mcts import policy_guided_mcts
from alphatensor.model import (
    load_policy_value_checkpoint,
    make_data_loader,
    make_sparse_policy_loader,
    run_epoch,
    run_sparse_policy_epoch,
)
from alphatensor.policy_replay import PolicyTargetReplayBuffer, transform_policy_target
from alphatensor.search import policy_guided_beam_search
from alphatensor.synthetic import generate_synthetic_dataset
from alphatensor.symmetry import random_basis_transform
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/mcts_self_play_transformer.pt"),
    )
    parser.add_argument(
        "--solutions", type=Path, default=Path("results/repeatability.json")
    )
    parser.add_argument(
        "--output-checkpoint", type=Path,
        default=Path("checkpoints/expert_bootstrap_transformer.pt"),
    )
    parser.add_argument(
        "--expert-replay-output", type=Path,
        default=Path("replay/expert_policy_replay.npz"),
    )
    parser.add_argument(
        "--history-output", type=Path,
        default=Path("results/expert_bootstrap_history.json"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--solution-limit", type=int, default=3,
        help="number of exact solution families to train on (use 1 for a coherent curriculum)",
    )
    parser.add_argument("--expert-passes", type=int, default=4)
    parser.add_argument("--symmetry-augmentations", type=int, default=16)
    parser.add_argument("--base-weight", type=float, default=8.0)
    parser.add_argument("--synthetic-games", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--beam-top-k", type=int, default=64)
    parser.add_argument("--mcts-simulations", type=int, default=3_000)
    parser.add_argument("--mcts-top-k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def beam_summary(result: object) -> dict[str, object]:
    residual = result.residual  # type: ignore[attr-defined]
    return {
        "success": bool(result.success),  # type: ignore[attr-defined]
        "action_indices": list(result.action_indices),  # type: ignore[attr-defined]
        "depth": int(result.depth),  # type: ignore[attr-defined]
        "expanded_nodes": int(result.expanded_nodes),  # type: ignore[attr-defined]
        "residual_nonzero": int(np.count_nonzero(residual)),
        "residual_rank_upper_bound": slice_rank_upper_bound(residual),
    }


def evaluate_beam(
    model: torch.nn.Module,
    target: np.ndarray,
    catalog: GF2ActionSpace,
    args: argparse.Namespace,
    device: str,
) -> object:
    # exact_tail_steps=0 is deliberate: success must come from model-guided actions.
    return policy_guided_beam_search(
        model,
        target,
        action_space=catalog,
        max_steps=7,
        beam_width=args.beam_width,
        top_k=args.beam_top_k,
        exact_tail_steps=0,
        seed=args.seed,
        device=device,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: dict[str, object],
    epoch: int,
    replay_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": source["architecture"],
            "model_config": source["model_config"],
            "source_checkpoint": str(source.get("source_checkpoint", "mcts_self_play")),
            "expert_bootstrap_epoch": epoch,
            "expert_replay_size": replay_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.expert_passes, args.synthetic_games, args.solution_limit) < 1:
        raise ValueError(
            "epochs, expert-passes, synthetic-games, and solution-limit must be positive"
        )
    if args.symmetry_augmentations < 0 or args.base_weight <= 0:
        raise ValueError("invalid augmentation count or base weight")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    solutions = load_exact_solutions(args.solutions)[: args.solution_limit]
    replay = PolicyTargetReplayBuffer(capacity=100_000, max_actions_per_state=32)

    for solution in solutions:
        targets = generate_expert_policy_targets(target, solution, action_space=catalog)
        for policy_target in targets:
            replay.add(policy_target, weight=args.base_weight)
        for _ in range(args.symmetry_augmentations):
            transform = random_basis_transform(rng)
            for policy_target in targets:
                replay.add(
                    transform_policy_target(
                        policy_target, transform, action_space=catalog
                    )
                )

    args.expert_replay_output.parent.mkdir(parents=True, exist_ok=True)
    replay.save(args.expert_replay_output)
    expert_arrays = replay.arrays()
    synthetic = generate_synthetic_dataset(
        args.synthetic_games, ranks=[5, 6, 7], seed=args.seed, action_space=catalog
    )
    validation = generate_synthetic_dataset(
        1_000, ranks=[5, 6, 7], seed=43, action_space=catalog
    )
    validation_loader = make_data_loader(
        validation, batch_size=args.batch_size, shuffle=False, seed=43
    )

    model, source = load_policy_value_checkpoint(args.checkpoint, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, object]] = []
    best_key: tuple[int, int, int, float, float] | None = None

    baseline = evaluate_beam(model, target, catalog, args, device)
    print(
        f"device={device} solutions={len(solutions)} expert_states={len(replay)} "
        f"baseline_success={baseline.success}"
    )
    for epoch in range(1, args.epochs + 1):
        synthetic_loader = make_data_loader(
            synthetic, batch_size=args.batch_size, shuffle=True,
            seed=args.seed + epoch,
        )
        synthetic_metrics = run_epoch(
            model, synthetic_loader, optimizer=optimizer, device=device
        )
        expert_metrics = None
        for pass_number in range(args.expert_passes):
            expert_loader = make_sparse_policy_loader(
                expert_arrays, batch_size=args.batch_size, shuffle=True,
                seed=args.seed + epoch * 100 + pass_number,
            )
            expert_metrics = run_sparse_policy_epoch(
                model, expert_loader, optimizer=optimizer, device=device
            )
        assert expert_metrics is not None
        validation_metrics = run_epoch(model, validation_loader, device=device)
        beam = evaluate_beam(model, target, catalog, args, device)
        summary = beam_summary(beam)
        key = (
            int(beam.success),
            -int(summary["residual_rank_upper_bound"]),
            -int(summary["residual_nonzero"]),
            float(expert_metrics.policy_top5_accuracy),
            -float(validation_metrics.loss),
        )
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(
                args.output_checkpoint,
                model=model,
                optimizer=optimizer,
                source=source,
                epoch=epoch,
                replay_size=len(replay),
            )
        record = {
            "epoch": epoch,
            "synthetic_loss": synthetic_metrics.loss,
            "expert_loss": expert_metrics.loss,
            "expert_top5": expert_metrics.policy_top5_accuracy,
            "validation_loss": validation_metrics.loss,
            "validation_top1": validation_metrics.policy_accuracy,
            "validation_top5": validation_metrics.policy_top5_accuracy,
            "beam": summary,
        }
        history.append(record)
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch} expert_loss={expert_metrics.loss:.4f} "
            f"expert_top5={expert_metrics.policy_top5_accuracy:.3%} "
            f"validation_top5={validation_metrics.policy_top5_accuracy:.3%} "
            f"beam_success={beam.success} residual_nonzero={summary['residual_nonzero']}"
        )
        if beam.success:
            break

    best_model, _ = load_policy_value_checkpoint(args.output_checkpoint, device=device)
    final_beam = evaluate_beam(best_model, target, catalog, args, device)
    mcts = policy_guided_mcts(
        best_model,
        target,
        action_space=catalog,
        max_steps=7,
        simulations=args.mcts_simulations,
        top_k=args.mcts_top_k,
        root_noise_fraction=0.0,
        exact_tail_steps=0,
        seed=args.seed,
        device=device,
    )
    selected = final_beam.action_indices if final_beam.success else mcts.action_indices
    factors = tuple(catalog.action(index) for index in selected)
    exact = len(selected) == 7 and verify_decomposition(target, factors)
    exhaustive_exact, checked = exhaustive_verify_2x2(factors) if exact else (False, 0)
    final = {
        "training_source": f"{len(solutions)} solver-discovered rank-7 solution family",
        "solver_used_during_evaluation": False,
        "exact_tail_completion_used": False,
        "expert_states": len(replay),
        "beam": beam_summary(final_beam),
        "mcts": {
            "success": mcts.success,
            "action_indices": list(mcts.action_indices),
            "simulations": mcts.simulations,
            "expanded_nodes": mcts.expanded_nodes,
            "residual_nonzero": mcts.residual_nonzero,
            "residual_rank_upper_bound": mcts.residual_rank_upper_bound,
        },
        "selected_exact_rank7": exact,
        "exhaustive_exact": exhaustive_exact,
        "checked_input_pairs": checked,
    }
    history.append({"final_evaluation": final})
    args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"replay={args.expert_replay_output.resolve()}")
    print(f"history={args.history_output.resolve()}")


if __name__ == "__main__":
    main()
