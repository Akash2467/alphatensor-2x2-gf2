# Exhaustive 5-to-4 Repair Results

Step 18 extended the exact local-repair engine from replacements of at most four
removed terms to five removed terms replaced by four. Four replacement actions
are represented as two unordered action pairs. A sorted meet-in-the-middle index
contains every pair from the complete 3,375-action catalog.

## Exact search structure

- Unordered action pairs: 5,693,625
- Distinct pair XOR codes: 4,964,625
- Maximum code multiplicity: 21
- Compact index memory: approximately 65.2 MB
- Complement lookup: chunked vectorized binary search
- Global distinctness and kept-action exclusions are enforced
- Every accepted result is checked by the exact GF(2) verifier

The constructed regression test successfully performs a genuine 5-to-4 repair
and produces an exactly verified rank-7 decomposition.

## Learned neighborhoods searched

1. `[274, 2081, 16, 2197, 679, 900, 339, 697]`
2. `[495, 60, 16, 678, 2497, 1717, 331, 1623]`
3. `[274, 2073, 331, 712, 686, 16, 1687, 0]`

For each neighborhood, all 56 possible five-action subsets were searched against
the full pair-pair index. No 5-to-4 replacement exists.

Combined with the earlier 2-to-1, 3-to-2, and 4-to-3 searches, all 210 subsets
that remove between two and five actions have now been checked for each rank-8
solution: 630 exact subset searches in total. Therefore, no rank-7 decomposition
can be reached from these neighborhoods while retaining three or more original
terms.

## Artifacts

- First result: `results/rank8_1_repair_5_to_4.json`
- Second result: `results/rank8_2_repair_5_to_4.json`
- Third result: `results/rank8_3_repair_5_to_4.json`
- Complete automated suite: 52 tests passing

## Next milestone

Local neighborhoods through five removed actions are exhausted. The next useful
step is a global exact rank-7 constraint solver that does not require overlap
with the learned rank-8 solutions. Its solution can be compared against the
agent's action probabilities and used as an independent evaluation oracle.
