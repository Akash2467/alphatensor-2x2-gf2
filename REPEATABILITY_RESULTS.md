# Repeatability Results

Step 20 repeated both global exact solving and stochastic policy-guided MCTS for
three seeds. Every exact solution was independently verified by tensor
reconstruction and all 256 binary matrix input pairs.

| Seed | Solver time | Rank-7 actions | Exact | MCTS result |
|-----:|------------:|---|:---:|---|
| 0 | 75.020 s | `[24, 332, 1117, 1646, 2479, 2888, 3315]` | Yes | Residual rank 1 |
| 1 | 89.858 s | `[137, 564, 686, 1624, 2235, 2408, 3187]` | Yes | Residual rank 1 |
| 2 | 79.704 s | `[436, 484, 704, 962, 1446, 1687, 3198]` | Yes | Residual rank 1 |

## Summary

- Exact solver success: 3 / 3 seeds
- Distinct exact rank-7 solutions: 3
- Exact tensor verification: 3 / 3
- Exhaustive 256-pair verification: 3 / 3
- Policy-guided MCTS rank-7 success: 0 / 3
- Best MCTS residual rank: 1
- Complete project tests: 57 passing

The global discovery component is reproducible. The reinforcement-learning
policy's limitation is also reproducible: it consistently finds an exact rank-8
completion but not the complete rank-7 path.

Machine-readable results: `results/repeatability.json`.
