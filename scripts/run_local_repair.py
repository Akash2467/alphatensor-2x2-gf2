"""Try to shorten an exact decomposition with bounded subset repair."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from alphatensor.actions import GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.repair import local_repair_decomposition
from alphatensor.tensor import matrix_multiplication_tensor


LEARNED_EIGHT_ACTIONS = [274, 2081, 16, 2197, 679, 900, 339, 697]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", type=int, default=LEARNED_EIGHT_ACTIONS)
    parser.add_argument("--max-removed", type=int, default=4)
    parser.add_argument("--min-removed", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("results/local_repair.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = matrix_multiplication_tensor(2)
    action_space = GF2ActionSpace()
    result = local_repair_decomposition(
        target,
        args.actions,
        min_removed=args.min_removed,
        max_removed=args.max_removed,
        action_space=action_space,
    )
    exact = verify_decomposition(
        target, [action_space.action(index) for index in result.repaired_actions]
    )
    payload = asdict(result) | {
        "repaired_rank": len(result.repaired_actions),
        "exact_verification": exact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
