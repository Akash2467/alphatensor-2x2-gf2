# MCTS Self-Play Fine-tuning Results

Step 15 closed the MCTS learning loop. Each cycle runs noisy MCTS searches,
retains diverse terminal trajectories, applies rank-preserving GF(2) symmetry
augmentation, adds multiple valid actions per state to replay, fine-tunes the
transformer on mixed synthetic/replay data, and evaluates deterministic MCTS.

## Configuration

- Starting checkpoint: `checkpoints/symmetry_transformer.pt`
- Starting replay records: 22,180
- Cycles: 2
- MCTS runs per cycle: 3
- Simulations per collection run: 3,000
- Trajectories retained per run: 64
- Symmetry variants per trajectory: 2
- Fine-tuning epochs per cycle: 2
- Deterministic evaluation simulations: 5,000

| Cycle | Replay records | Validation top-1 | Validation top-5 | Evaluation residual |
|------:|---------------:|-----------------:|-----------------:|--------------------:|
| 1 | 24,918 | 22.409% | 29.696% | 1 rank-one term |
| 2 | 27,621 | 22.426% | 29.829% | 1 rank-one term |

## Outcome

- Replay gained 5,441 deduplicated MCTS/symmetry records.
- Validation top-5 improved from 29.762% to 29.829% overall.
- The saved 330,096-parameter checkpoint and replay archive reload correctly.
- A final 10,000-simulation, top-256 exploratory search expanded 1,138 nodes.
- The best result still leaves one legal rank-one residual after seven actions,
  so no exact rank-7 decomposition was discovered.
- Complete automated suite: 43 tests passing.

## Artifacts

- Checkpoint: `checkpoints/mcts_self_play_transformer.pt`
- Replay: `replay/mcts_replay.npz`
- Cycle history: `results/mcts_self_play_history.json`
- Final exploratory search: `results/mcts_self_play_exploratory.json`

## Next milestone

Visit-count distributions and batched multi-tree evaluation were completed in
Step 16. The next milestone is outcome-weighted filtering so weak MCTS trees do
not dilute the stronger policy learned in Step 15.
