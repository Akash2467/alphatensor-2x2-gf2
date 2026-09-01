"""Exhaustively verify and export a solved rank-7 decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphatensor.actions import GF2ActionSpace
from alphatensor.algorithm import describe_equations, exhaustive_verify_2x2
from alphatensor.decomposition import verify_decomposition
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, default=Path("results/rank7_smt.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/rank7_verification.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solution = json.loads(args.solution.read_text(encoding="utf-8"))
    action_indices = tuple(int(index) for index in solution["action_indices"])
    action_space = GF2ActionSpace()
    factors = [action_space.action(index) for index in action_indices]
    tensor_exact = verify_decomposition(matrix_multiplication_tensor(2), factors)
    exhaustive_exact, checked_pairs = exhaustive_verify_2x2(factors)
    payload = {
        "action_indices": list(action_indices),
        "rank": len(action_indices),
        "tensor_exact": tensor_exact,
        "exhaustive_exact": exhaustive_exact,
        "checked_input_pairs": checked_pairs,
        "equations": describe_equations(factors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
