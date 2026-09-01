"""Globally solve and analyze a rank-7 decomposition over GF(2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.constraint import solve_exact_rank
from alphatensor.model import load_policy_value_checkpoint
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mcts_self_play_transformer.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/rank7_smt.json"))
    return parser.parse_args()


@torch.no_grad()
def sequential_policy_analysis(
    checkpoint: Path,
    target: np.ndarray,
    actions: tuple[int, ...],
    action_space: GF2ActionSpace,
) -> list[dict[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_policy_value_checkpoint(checkpoint, device=device)
    residual = target.copy()
    remaining = set(actions)
    analysis: list[dict[str, object]] = []
    while remaining:
        state = torch.from_numpy(residual).to(device=device, dtype=torch.float32)
        logits, _ = model(state.mul(2.0).sub(1.0).unsqueeze(0))
        logits = logits[0]
        ordering = torch.argsort(logits, descending=True)
        ranks = torch.empty_like(ordering)
        ranks[ordering] = torch.arange(len(ordering), device=device)
        chosen = max(remaining, key=lambda action: float(logits[action]))
        probabilities = torch.softmax(logits, dim=0)
        analysis.append(
            {
                "step": len(analysis) + 1,
                "action_index": chosen,
                "global_policy_rank": int(ranks[chosen]) + 1,
                "policy_probability": float(probabilities[chosen]),
            }
        )
        residual ^= action_space.term(chosen)
        remaining.remove(chosen)
    return analysis


def main() -> None:
    args = parse_args()
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    result = solve_exact_rank(
        target,
        args.rank,
        action_space=action_space,
        timeout_seconds=args.timeout_seconds,
        random_seed=args.seed,
    )
    factors = [action_space.action(index) for index in result.action_indices]
    policy_analysis = (
        sequential_policy_analysis(
            args.checkpoint, target, result.action_indices, action_space
        )
        if result.status == "sat"
        else []
    )
    payload = {
        "status": result.status,
        "rank": result.rank,
        "action_indices": list(result.action_indices),
        "elapsed_seconds": result.elapsed_seconds,
        "exact_verification": result.exact_verification,
        "reason_unknown": result.reason_unknown,
        "factors": [
            {"u": u.tolist(), "v": v.tolist(), "w": w.tolist()}
            for u, v, w in factors
        ],
        "policy_analysis": policy_analysis,
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
