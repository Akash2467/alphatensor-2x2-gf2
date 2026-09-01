# Visit-Distribution Training and Batched MCTS Results

Step 16 replaced repeated hard MCTS labels with sparse visit-count probability
distributions. Repeated states merge their visit counts, and the transformer is
trained with cross-entropy against all supported actions. Independent MCTS trees
evaluate one leaf each in a shared transformer batch.

## Implementation

- Sparse policy distributions retain up to 16 actions per state.
- Visit counts from repeated states and runs are merged and normalized.
- GF(2) basis symmetry transforms both states and every supported action.
- Hard synthetic/replay epochs alternate with soft-policy distribution epochs.
- Four independently noised trees share batched model evaluation.

## Experiment

- Starting checkpoint: `checkpoints/mcts_self_play_transformer.pt`
- Cycles: 2
- Batched trees per collection: 4
- Collections per cycle: 2
- Simulations per tree: 1,000
- Final policy-distribution replay: 4,262 states

| Cycle | Policy states | Validation top-1 | Validation top-5 | Residual after 7 actions |
|------:|--------------:|-----------------:|-----------------:|-------------------------:|
| 1 | 2,302 | 22.592% | 29.696% | 1 rank-one term |
| 2 | 4,262 | 22.658% | 29.679% | 1 rank-one term |

## Outcome

- Batched collection and soft-policy training work end to end.
- The distribution-trained checkpoint reloads correctly.
- Validation did not beat the 29.829% starting checkpoint, so
  `checkpoints/mcts_self_play_transformer.pt` remains the project's best model.
- The experimental checkpoint found a third exact rank-8 neighborhood:
  `[274, 2073, 331, 712, 686, 16, 1687, 0]`.
- Exact local repair checked all 154 subsets through 4-to-3 replacement and found
  no rank-7 compression.
- Exact rank 7 remains undiscovered.
- Complete automated suite: 47 tests passing.

## Artifacts

- Experimental checkpoint: `checkpoints/distribution_transformer.pt`
- Policy replay: `replay/policy_distribution_replay.npz`
- History: `results/distribution_history.json`
- Local repair: `results/distribution_local_repair.json`

## Next milestone

Outcome weighting, weak-tree filtering, rollback, and best-checkpoint retention
were completed in Step 17. The next milestone is a broader exact 5-to-4 repair
search across all learned rank-8 neighborhoods.
