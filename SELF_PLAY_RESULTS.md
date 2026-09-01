# Target Replay and Fine-tuning Results

The transformer completed three target-specific replay/fine-tuning cycles. Each
cycle ran four noisy policy-guided searches, retained useful target trajectories,
mixed replay states with synthetic training data, and evaluated a deterministic
beam search from the real 2x2 multiplication tensor.

| Cycle | Replay states | Validation top-1 | Validation top-5 | Best residual nonzeros |
|------:|--------------:|-----------------:|-----------------:|-----------------------:|
| 1 | 279 | 20.612% | 28.315% | 3 |
| 2 | 409 | 20.745% | 28.564% | 2 |
| 3 | 464 | 20.978% | 28.764% | 1 |

## Outcome

- The run completed successfully and saved a loadable 330,096-parameter
  transformer checkpoint.
- Search improved from a four-entry residual before replay to a one-entry
  residual after replay.
- A one-entry binary tensor is rank one, so appending it to the selected seven
  actions produces an exact eight-multiplication decomposition.
- The independently verified eight-action sequence is
  `[274, 2081, 16, 2197, 679, 900, 339, 697]`.
- The final completion action has factors `u=[0,1,0,0]`,
  `v=[0,0,1,0]`, and `w=[1,0,0,0]` and removes residual coordinate
  `[1,2,0]`.
- No exact seven-action decomposition was found, so the final rank-7 discovery
  and verification stages remain incomplete.

## Artifacts

- Checkpoint: `checkpoints/self_play_transformer.pt`
- Replay buffer: `replay/target_replay.npz`
- Per-cycle history: `results/self_play_history.json`

## Next milestone

Add invertible-basis symmetry augmentation and replace single-action imitation
targets with search-derived multi-action policy targets. Then repeat target
replay and wider search to try to remove the final rank-one residual within the
seven-action budget.
