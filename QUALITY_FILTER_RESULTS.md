# Outcome-Weighted Filtering and Checkpoint Retention Results

Step 17 prevents weak MCTS distributions from replacing a stronger model. Search
results receive an exponential weight based on residual rank/nonzero count,
trees above a configurable quality threshold are discarded, and candidate
checkpoints are compared lexicographically by exact success, residual rank,
residual nonzeros, and fixed validation top-5 accuracy.

If a cycle does not improve the incumbent, both model and optimizer are rolled
back before the next cycle. The final output always contains the best selected
state rather than simply the most recent state.

## Controlled experiment

- Incumbent checkpoint: `checkpoints/mcts_self_play_transformer.pt`
- Incumbent validation top-5: 29.829%
- Accepted residual-rank threshold: 1
- Quality temperature: 1.0
- Cycles: 2
- Final quality-filtered policy replay: 4,303 states

| Cycle | Validation top-5 | Target residual | Selected? |
|------:|-----------------:|----------------:|:---------:|
| Baseline | 29.829% | 1 rank-one term | Yes |
| 1 | 29.795% | 1 rank-one term | No |
| 2 | 29.213% | 1 rank-one term | No |

## Outcome

- Both regressing candidates were rejected automatically.
- Cycle 2 restarted from the restored incumbent rather than Cycle 1's weights.
- The final saved checkpoint reports selected cycle `0`.
- Every saved model parameter is exactly equal to the incumbent parameter.
- The replay archive remains available for later curriculum experiments.
- Exact rank 7 was not discovered.
- Complete automated suite: 51 tests passing.

## Artifacts

- Preserved-best checkpoint: `checkpoints/quality_filtered_transformer.pt`
- Quality replay: `replay/quality_policy_replay.npz`
- Selection history: `results/quality_filtered_history.json`

## Next milestone

The learning pipeline is protected against regression, and Step 18 subsequently
exhausted exact replacements involving five removed terms across all three
rank-8 neighborhoods. The next milestone is a global rank-7 constraint solver.
