# Symmetry-Augmented Multi-Action Training Results

Step 12 added independent invertible GF(2) basis transformations on all three
tensor modes. These transformations preserve tensor rank and map every action to
another legal action in the 3,375-element catalog. Replay now retains up to eight
high-quality next actions for the same state instead of discarding all but one.

## Experiment

- Starting checkpoint: `checkpoints/self_play_transformer.pt`
- Cycles: 3
- Noisy searches per cycle: 4
- Symmetry variants per retained trajectory: 4
- Maximum actions retained per state: 8
- Final replay records: 22,180
- Fine-tuning epochs per cycle: 2
- Evaluation beam width/top-k: 1,024 / 128

| Cycle | Replay records | Validation top-1 | Validation top-5 | Evaluation residual nonzeros |
|------:|---------------:|-----------------:|-----------------:|-----------------------------:|
| 1 | 7,766 | 21.877% | 29.379% | 1 |
| 2 | 14,977 | 21.926% | 29.596% | 1 |
| 3 | 22,180 | 22.342% | 29.762% | 1 |

## Outcome

- Validation top-5 improved from 28.764% before Step 12 to 29.762%.
- Validation top-1 improved from 20.978% to 22.342%.
- The saved 330,096-parameter checkpoint and replay archive reload correctly.
- The best seven-action trajectory still leaves one rank-one residual, so the
  previously verified eight-multiplication solution remains the best discovery.
- No exact rank-7 decomposition was discovered in this experiment.
- The complete automated suite passes: 39 tests.

## Artifacts

- Checkpoint: `checkpoints/symmetry_transformer.pt`
- Replay buffer: `replay/symmetry_replay.npz`
- Per-cycle history: `results/symmetry_history.json`

## Next milestone

The exact local repair search was completed and found no shortening replacement
involving four or fewer removed terms. The next milestone is policy-guided Monte
Carlo tree search for broader exploration outside that local neighborhood.
