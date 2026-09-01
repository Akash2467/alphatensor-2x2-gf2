"""Repeat exact rank-7 solving and policy-guided MCTS across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.algorithm import exhaustive_verify_2x2
from alphatensor.constraint import solve_exact_rank
from alphatensor.mcts import policy_guided_mcts
from alphatensor.model import load_policy_value_checkpoint
from alphatensor.repeatability import summarize_repeatability
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--solver-timeout", type=float, default=180.0)
    parser.add_argument("--mcts-simulations", type=int, default=2_000)
    parser.add_argument("--mcts-top-k", type=int, default=256)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mcts_self_play_transformer.pt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/repeatability.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be distinct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_policy_value_checkpoint(args.checkpoint, device=device)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    runs: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        solver_result = solve_exact_rank(
            target,
            7,
            action_space=action_space,
            timeout_seconds=args.solver_timeout,
            random_seed=seed,
        )
        exhaustive_exact = False
        checked_pairs = 0
        if solver_result.status == "sat":
            factors = [
                action_space.action(index) for index in solver_result.action_indices
            ]
            exhaustive_exact, checked_pairs = exhaustive_verify_2x2(factors)

        mcts_result = policy_guided_mcts(
            model,
            target,
            action_space=action_space,
            simulations=args.mcts_simulations,
            top_k=args.mcts_top_k,
            c_puct=3.0,
            value_blend=0.25,
            root_noise_fraction=0.5,
            dirichlet_alpha=0.15,
            exact_tail_steps=2,
            seed=seed,
            device=device,
        )
        run = {
            "seed": seed,
            "solver_status": solver_result.status,
            "solver_seconds": solver_result.elapsed_seconds,
            "solver_action_indices": list(solver_result.action_indices),
            "solver_exact": solver_result.exact_verification,
            "exhaustive_exact": exhaustive_exact,
            "checked_input_pairs": checked_pairs,
            "mcts_success": mcts_result.success,
            "mcts_action_indices": list(mcts_result.action_indices),
            "mcts_expanded_nodes": mcts_result.expanded_nodes,
            "mcts_residual_nonzero": mcts_result.residual_nonzero,
            "mcts_residual_rank": mcts_result.residual_rank_upper_bound,
        }
        runs.append(run)
        payload = {"runs": runs, "summary": summarize_repeatability(runs)}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"seed={seed} solver={solver_result.status} "
            f"seconds={solver_result.elapsed_seconds:.3f} "
            f"mcts_success={mcts_result.success} "
            f"mcts_residual_rank={mcts_result.residual_rank_upper_bound}"
        )

    print(json.dumps(summarize_repeatability(runs), indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
