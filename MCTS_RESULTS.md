# Policy-Guided Monte Carlo Tree Search Results

Step 14 added PUCT Monte Carlo tree search guided by the transformer's policy
priors and value estimate. Root Dirichlet noise provides exploration, repeated
actions are excluded, and exact one/two-action completion is used near the
seven-action limit.

## Validation

- A scripted-policy test discovers and exactly verifies a rank-7 decomposition.
- Invalid search configurations are rejected.
- Complete project suite: 43 tests passing.

## Experiments

| Configuration | Simulations | Top-k | Expanded nodes | Best residual | Rank-7 found |
|---|---:|---:|---:|---:|---:|
| Standard PUCT | 5,000 | 128 | 342 | 1 rank-one term | No |
| High exploration | 10,000 | 256 | 1,076 | 1 rank-one term | No |

The standard run returned the previously known near-solution. The exploratory
run reached a different seven-action path:

`[495, 60, 16, 678, 2497, 1717, 331]`

Its residual is legal action `1623`, producing the independently verified exact
rank-8 decomposition:

`[495, 60, 16, 678, 2497, 1717, 331, 1623]`

Exact local repair subsequently checked all 154 subsets involving at most four
removed terms from this second solution. No rank-7 repair exists in that local
neighborhood either.

## Artifacts

- Standard result: `results/mcts_search.json`
- Exploratory result: `results/mcts_exploratory.json`
- Repair result for the new neighborhood: `results/mcts_local_repair.json`

## Next milestone

This MCTS replay/fine-tuning loop was completed in Step 15. The next milestone is
training on visit-count policy distributions and batching leaf evaluation to
increase both target quality and search coverage.
