"""Evaluate rank-7 policy robustness without solver or exact-tail assistance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.algorithm import exhaustive_verify_2x2
from alphatensor.decomposition import verify_decomposition
from alphatensor.expert import load_exact_solutions
from alphatensor.model import load_policy_value_checkpoint
from alphatensor.search import policy_guided_beam_search
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/expert_bootstrap_transformer.pt"),
    )
    parser.add_argument(
        "--solutions", type=Path, default=Path("results/repeatability.json")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/expert_robustness.json"),
    )
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--noise-seeds", type=int, default=5)
    return parser.parse_args()


def classify_solution(
    actions: tuple[int, ...], known: tuple[tuple[int, ...], ...]
) -> str:
    action_set = frozenset(actions)
    for number, solution in enumerate(known, start=1):
        if action_set == frozenset(solution):
            return f"known_solution_{number}"
    return "novel" if len(actions) == 7 else "incomplete"


def run_search(
    model: torch.nn.Module,
    target: np.ndarray,
    catalog: GF2ActionSpace,
    known: tuple[tuple[int, ...], ...],
    *,
    beam_width: int,
    top_k: int,
    noise: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    result = policy_guided_beam_search(
        model,
        target,
        action_space=catalog,
        max_steps=7,
        beam_width=beam_width,
        top_k=top_k,
        exact_tail_steps=0,
        policy_noise=noise,
        seed=seed,
        device=device,
    )
    factors = tuple(catalog.action(index) for index in result.action_indices)
    tensor_exact = len(factors) == 7 and verify_decomposition(target, factors)
    exhaustive_exact, checked = (
        exhaustive_verify_2x2(factors) if tensor_exact else (False, 0)
    )
    return {
        "beam_width": beam_width,
        "top_k": top_k,
        "policy_noise": noise,
        "seed": seed,
        "success": result.success,
        "action_indices": list(result.action_indices),
        "solution_class": (
            classify_solution(result.action_indices, known)
            if tensor_exact
            else "incomplete"
        ),
        "expanded_nodes": result.expanded_nodes,
        "residual_nonzero": int(np.count_nonzero(result.residual)),
        "tensor_exact": tensor_exact,
        "exhaustive_exact": exhaustive_exact,
        "checked_input_pairs": checked,
    }


def main() -> None:
    args = parse_args()
    if any(width < 1 for width in args.beam_widths):
        raise ValueError("beam widths must be positive")
    if args.top_k < 1 or args.noise < 0 or args.noise_seeds < 1:
        raise ValueError("invalid top-k, noise, or seed count")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_policy_value_checkpoint(args.checkpoint, device=device)
    catalog = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    known = load_exact_solutions(args.solutions)
    runs: list[dict[str, object]] = []

    for width in args.beam_widths:
        record = run_search(
            model, target, catalog, known, beam_width=width, top_k=args.top_k,
            noise=0.0, seed=0, device=device,
        )
        runs.append(record)
        print(
            f"deterministic width={width} success={record['success']} "
            f"residual={record['residual_nonzero']} class={record['solution_class']}"
        )

    robust_width = max(args.beam_widths)
    for seed in range(args.noise_seeds):
        record = run_search(
            model, target, catalog, known, beam_width=robust_width,
            top_k=args.top_k, noise=args.noise, seed=seed, device=device,
        )
        runs.append(record)
        print(
            f"noisy seed={seed} success={record['success']} "
            f"residual={record['residual_nonzero']} class={record['solution_class']}"
        )

    deterministic = [run for run in runs if run["policy_noise"] == 0.0]
    noisy = [run for run in runs if run["policy_noise"] != 0.0]
    successful = [run for run in runs if run["success"]]
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "solver_used": False,
        "exact_tail_completion_used": False,
        "deterministic_successes": sum(bool(run["success"]) for run in deterministic),
        "deterministic_runs": len(deterministic),
        "minimum_successful_beam_width": min(
            (int(run["beam_width"]) for run in deterministic if run["success"]),
            default=None,
        ),
        "noisy_successes": sum(bool(run["success"]) for run in noisy),
        "noisy_runs": len(noisy),
        "all_successes_tensor_exact": all(bool(run["tensor_exact"]) for run in successful),
        "all_successes_exhaustive_exact": all(
            bool(run["exhaustive_exact"]) for run in successful
        ),
        "solution_classes": sorted({str(run["solution_class"]) for run in successful}),
    }
    payload = {"summary": summary, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
