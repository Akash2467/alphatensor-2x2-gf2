"""Run transformer-guided beam search on the 2 x 2 multiplication tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alphatensor.actions import GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.game import slice_rank_upper_bound
from alphatensor.model import load_policy_value_checkpoint
from alphatensor.search import policy_guided_beam_search
from alphatensor.tensor import matrix_multiplication_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/transformer_policy_value.pt"),
    )
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--value-weight", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--exact-tail-steps", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--output", type=Path, default=Path("results/beam_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_policy_value_checkpoint(args.checkpoint, device=device)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    result = policy_guided_beam_search(
        model,
        target,
        action_space=action_space,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        top_k=args.top_k,
        value_weight=args.value_weight,
        exact_tail_steps=args.exact_tail_steps,
        device=device,
    )

    factors = [action_space.action(index) for index in result.action_indices]
    exact = result.success and verify_decomposition(target, factors)
    residual_nonzero = int(torch.from_numpy(result.residual).count_nonzero())
    payload = {
        "success": result.success,
        "exact": exact,
        "depth": result.depth,
        "expanded_nodes": result.expanded_nodes,
        "final_beam_size": result.final_beam_size,
        "score": result.score,
        "residual_nonzero": residual_nonzero,
        "residual_rank_upper_bound": slice_rank_upper_bound(result.residual),
        "action_indices": list(result.action_indices),
        "factors": [
            {
                "u": factor[0].tolist(),
                "v": factor[1].tolist(),
                "w": factor[2].tolist(),
            }
            for factor in factors
        ],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": checkpoint.get("architecture", "mlp"),
        "beam_width": args.beam_width,
        "top_k": args.top_k,
        "value_weight": args.value_weight,
        "exact_tail_steps": args.exact_tail_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
