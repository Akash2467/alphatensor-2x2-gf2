# Exact Local Repair Results

Step 13 implemented an exact subset-replacement search around the learned
eight-action decomposition:

`[274, 2081, 16, 2197, 679, 900, 339, 697]`

The repair engine tries to shorten a valid decomposition by removing a subset
and reconstructing the removed tensor contribution with one fewer rank-one
term. Every accepted candidate is independently checked by the exact GF(2)
decomposition verifier.

## Searches performed

- Remove 2 actions and replace them with 1.
- Remove 3 actions and replace them with 2.
- Remove 4 actions and replace them with 3.
- Total subsets checked: 154.
- Action catalog searched: all 3,375 legal rank-one actions.

The 4-to-3 search uses a compact meet-in-the-middle index over every unordered
pair of legal actions, so it remains exact rather than sampling candidates.

## Result

- No shortening repair was found.
- The returned decomposition remains rank 8.
- Exact verification of the original decomposition still passes.
- This proves that this particular learned solution cannot be converted to rank
  7 by changing four or fewer of its terms.
- Complete automated suite after MCTS integration: 43 tests passing.

After Step 14, the same exact repair search was applied to a second rank-8
decomposition found by high-exploration MCTS. Its 154 subsets also contained no
repair involving four or fewer removed terms.

## Artifact

- Machine-readable result: `results/local_repair.json`

## Next milestone

Implement policy-guided Monte Carlo tree search. Unlike the current beam search,
MCTS can retain and revisit diverse low-probability branches instead of pruning
them permanently at each depth, which may escape the rank-8 solution's local
neighborhood.
