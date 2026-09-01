"""Train the small policy-value model on synthetic demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alphatensor.model import (
    PolicyValueNetwork,
    TensorTransformerPolicyValueNetwork,
    make_data_loader,
    run_epoch,
)
from alphatensor.synthetic import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--validation-games", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument(
        "--architecture", choices=["transformer", "mlp"], default="transformer"
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--feedforward-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric", choices=["top5", "loss"], default="top5"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/policy_value.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games < 1 or args.validation_games < 1:
        raise ValueError("games and validation-games must be positive")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_data = generate_synthetic_dataset(
        args.games, ranks=[5, 6, 7], seed=args.seed
    )
    validation_data = generate_synthetic_dataset(
        args.validation_games, ranks=[5, 6, 7], seed=args.seed + 1
    )
    training_loader = make_data_loader(
        training_data,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    validation_loader = make_data_loader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
    )
    if args.architecture == "transformer":
        model = TensorTransformerPolicyValueNetwork(
            d_model=args.d_model,
            num_heads=args.attention_heads,
            num_layers=args.transformer_layers,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
        )
        model_config = {
            "d_model": args.d_model,
            "num_heads": args.attention_heads,
            "num_layers": args.transformer_layers,
            "feedforward_size": args.feedforward_size,
            "dropout": args.dropout,
        }
    else:
        model = PolicyValueNetwork(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.hidden_layers,
        )
        model_config = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.hidden_layers,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(
        f"device={device} architecture={args.architecture} "
        f"train_games={training_data.num_games} "
        f"train_samples={len(training_data)} validation_games={validation_data.num_games} "
        f"validation_samples={len(validation_data)}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_selection_value = (
        float("-inf") if args.selection_metric == "top5" else float("inf")
    )
    best_validation_loss = float("inf")
    best_validation_top5 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        training_metrics = run_epoch(
            model,
            training_loader,
            optimizer=optimizer,
            device=device,
            value_weight=args.value_weight,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device=device,
            value_weight=args.value_weight,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={training_metrics.loss:.4f} "
            f"train_top1={training_metrics.policy_accuracy:.3%} "
            f"train_top5={training_metrics.policy_top5_accuracy:.3%} "
            f"validation_loss={validation_metrics.loss:.4f} "
            f"validation_top1={validation_metrics.policy_accuracy:.3%} "
            f"validation_top5={validation_metrics.policy_top5_accuracy:.3%}"
        )

        selection_value = (
            validation_metrics.policy_top5_accuracy
            if args.selection_metric == "top5"
            else validation_metrics.loss
        )
        improved = (
            selection_value > best_selection_value
            if args.selection_metric == "top5"
            else selection_value < best_selection_value
        )
        if improved:
            best_selection_value = selection_value
            best_validation_loss = validation_metrics.loss
            best_validation_top5 = validation_metrics.policy_top5_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                    "best_validation_top5": best_validation_top5,
                    "selection_metric": args.selection_metric,
                    "seed": args.seed,
                    "training_games": args.games,
                    "validation_games": args.validation_games,
                    "architecture": args.architecture,
                    "model_config": model_config,
                },
                args.output,
            )
            print(f"best_checkpoint={args.output.resolve()}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stopping epoch={epoch} patience={args.patience}")
                break

    print(
        f"training_complete best_epoch={best_epoch} "
        f"best_validation_loss={best_validation_loss:.4f} "
        f"best_validation_top5={best_validation_top5:.3%}"
    )


if __name__ == "__main__":
    main()
