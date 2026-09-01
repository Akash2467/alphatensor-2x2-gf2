# Global Exact Rank-7 Solver Results

Step 19 formulated global rank-7 discovery over the complete finite action
catalog. One Boolean variable represents each of the 3,375 legal rank-one
actions. The constraints require exactly seven selected actions and impose one
XOR equation for each of the 64 multiplication-tensor coordinates.

No known decomposition, rank-8 neighborhood, or action overlap was supplied.
Z3's finite-domain SAT backend returned a solution in 117.692 seconds:

`[24, 332, 1117, 1646, 2479, 2888, 3315]`

## Verification

- Selected actions: exactly 7
- Exact 4 x 4 x 4 GF(2) tensor reconstruction: passed
- Exhaustive binary 2 x 2 matrix pairs: 256 / 256 passed
- Complete automated project suite: 56 tests passed

## Transformer comparison

The seven solution actions were ordered greedily by the trained transformer's
score among the remaining correct actions. Their global policy ranks were:

| Step | Action | Global policy rank | Probability |
|-----:|-------:|-------------------:|------------:|
| 1 | 24 | 71 | 0.001120 |
| 2 | 3315 | 232 | 0.000396 |
| 3 | 2479 | 575 | 0.000177 |
| 4 | 2888 | 248 | 0.000627 |
| 5 | 1117 | 3 | 0.043543 |
| 6 | 1646 | 1 | 0.327722 |
| 7 | 332 | 1 | 0.543088 |

The transformer recognizes the final portion strongly, but several early path
actions fall outside beam/MCTS candidate sets. This explains why the learned
agent repeatedly reached rank 8 while the global solver found rank 7.

## Artifacts

- Solver result: `results/rank7_smt.json`
- Exhaustive verification: `results/rank7_verification.json`
- Solver CLI: `scripts/solve_rank7.py`
- Verification CLI: `scripts/verify_rank7_solution.py`

## Scope note

The project independently discovered rank 7 globally, but the reinforcement-
learning policy did not generate the entire exact path unaided. The remaining
evaluation step is repeatability across solver and search seeds.
