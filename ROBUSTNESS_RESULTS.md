# Rank-7 Policy Robustness Evaluation

## Evaluation controls

- Checkpoint: `checkpoints/expert_bootstrap_transformer.pt`
- SAT/constraint solver: disabled
- Exact one/two-action tail completion: disabled
- Policy top-k: 64
- Maximum path length: 7 actions
- Every successful result checked by exact tensor reconstruction and all 256
  binary 2x2 matrix input pairs

## Deterministic beam-width sweep

| Beam width | Rank-7 success | Final nonzero residual entries |
|---:|:---:|---:|
| 32 | No | 3 |
| 64 | No | 3 |
| 128 | No | 3 |
| 256 | Yes | 0 |

The minimum tested successful width is 256. The model contains the correct
policy information, but a narrower search prunes the complete trajectory.

## Controlled-noise evaluation

At beam width 256, policy noise 0.02 was applied independently for seeds 0-4.

- Successful runs: 5/5
- Exact tensor checks: 5/5
- Exhaustive 256-pair checks: 5/5

This indicates stability under small perturbations once sufficient beam capacity
is available.

## Generalization finding

All six successful runs—the deterministic run and five noisy runs—used the same
seven-factor set as the expert training solution, although action order could
vary. None matched either held-out solver solution and no novel exact solution
was found.

Therefore the policy robustly reproduces its taught solution, but this experiment
does not show structural generalization to unseen rank-7 decompositions. The next
research target is self-play or diversity training that can produce a held-out or
novel exact factor set.

Machine-readable results are in `results/expert_robustness.json`.
