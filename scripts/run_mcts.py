"""Run transformer-guided Monte Carlo tree search on the real target tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.mcts import policy_guided_mcts
from alphatensor.model import load_policy_value_checkpoint
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/symmetry_transformer.pt"),
    )
    parser.add_argument("--simulations", type=int, default=5_000)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--value-blend", type=float, default=0.5)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mcts_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_policy_value_checkpoint(args.checkpoint, device=device)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    result = policy_guided_mcts(
        model,
        target,
        action_space=action_space,
        simulations=args.simulations,
        top_k=args.top_k,
        c_puct=args.c_puct,
        value_blend=args.value_blend,
        root_noise_fraction=args.root_noise_fraction,
        dirichlet_alpha=args.dirichlet_alpha,
        exact_tail_steps=2,
        seed=args.seed,
        device=device,
    )
    factors = [action_space.action(index) for index in result.action_indices]
    exact = result.success and verify_decomposition(target, factors)
    payload = {
        "success": result.success,
        "exact": exact,
        "action_indices": list(result.action_indices),
        "depth": len(result.action_indices),
        "simulations": result.simulations,
        "expanded_nodes": result.expanded_nodes,
        "root_visits": result.root_visits,
        "score": result.score,
        "residual_nonzero": result.residual_nonzero,
        "residual_rank_upper_bound": result.residual_rank_upper_bound,
        "checkpoint": str(args.checkpoint),
        "checkpoint_cycle": checkpoint.get(
            "distribution_self_play_cycle",
            checkpoint.get("mcts_self_play_cycle", checkpoint.get("self_play_cycle")),
        ),
        "top_k": args.top_k,
        "c_puct": args.c_puct,
        "value_blend": args.value_blend,
        "root_noise_fraction": args.root_noise_fraction,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
