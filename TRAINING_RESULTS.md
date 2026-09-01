# Supervised Training Results

## Transformer result (current model)

The current model is a compact transformer with:

- 64 tensor-entry tokens and one classification token
- Token dimension: 64
- Attention heads: 4
- Transformer encoder layers: 2
- Feed-forward size: 256
- Global 64-entry residual projection added to the classification representation
- Policy head: 3,375 actions
- Value head: remaining decomposition moves
- Parameters: 330,096

Training used 10,000 games (60,021 states) and validation used 1,000 games
(6,011 states). Checkpoints were selected using validation top-5 policy
accuracy because beam search consumes the highest-ranked actions.

- Maximum epochs: 20
- Early-stopping patience: 5
- Training stopped: epoch 16
- Best checkpoint: epoch 11
- Validation loss at selected checkpoint: 6.5481
- Validation top-1 policy accuracy: 19.797%
- Validation top-5 policy accuracy: 27.683%
- Checkpoint: `checkpoints/transformer_policy_value.pt`
- Checkpoint size: approximately 4.0 MB

The checkpoint was independently reloaded and produced the same metrics.

## MLP baseline result

### Configuration

- Training games: 10,000
- Validation games: 1,000
- Synthetic ranks: 5, 6 and 7
- Training samples: 60,021
- Validation samples: 6,011
- Maximum epochs: 20
- Batch size: 256
- Early-stopping patience: 5
- Random seed: 42
- Device: CPU

### Result

Training stopped after epoch 8 because validation loss did not improve for five
epochs. The saved model is the best-validation checkpoint from epoch 3.

- Validation loss: 6.4525
- Validation top-1 policy accuracy: 16.553%
- Validation top-5 policy accuracy: 24.854%
- Checkpoint: `checkpoints/policy_value.pt`
- Checkpoint size: approximately 12.6 MB

The checkpoint was independently reloaded and reevaluated against the original
validation seed. The reproduced metrics matched the training run.

## Comparison and interpretation

The transformer improved over the MLP baseline:

- Top-1: 19.797% versus 16.553%
- Top-5: 27.683% versus 24.854%
- The transformer checkpoint is approximately 4.0 MB versus 12.6 MB for the
  MLP checkpoint including optimizer state.

Random selection among 3,375 actions has approximately 0.030% top-1 accuracy
and 0.148% top-5 accuracy. The trained network is therefore learning useful
synthetic decomposition patterns. The next test is whether those patterns
transfer to the real 2x2 matrix-multiplication tensor through beam search.
