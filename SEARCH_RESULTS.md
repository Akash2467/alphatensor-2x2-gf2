# Beam Search Results

The trained transformer was tested on the actual 2x2 GF(2)
matrix-multiplication tensor. Search code was independently verified with a
scripted policy that finds and exactly verifies a seven-action decomposition.

## Experiments

### Standard search

- Beam width: 256
- Top-k actions per state: 32
- Value weight: 0.1
- Expanded nodes: 40,512
- Result: no exact rank-7 decomposition

### Wide search

- Beam width: 1,024
- Top-k actions per state: 128
- Value weight: 0.05
- Expanded nodes: 661,306
- Result: no exact rank-7 decomposition

### Wide hybrid search

- Beam width: 2,048
- Top-k actions per state: 256
- Value weight: 0 (pure policy ordering)
- Exact algebraic completion of one- and two-action tails
- Expanded nodes: 2,660,633
- Result: no exact rank-7 decomposition

All three searches selected the same highest-scoring seven-action sequence. Its
residual contains four nonzero entries and has a slice-rank upper bound of four,
so it is close but not an exact solution.

## Diagnosis

- The best known Strassen-compatible first action is ranked 117th by the model,
  outside the standard top-32 search.
- Even after top-k was increased to 256, the complete correct path did not
  survive beam pruning.
- The value head predicts about 3.1 moves for the initial target even though the
  target requires at least seven, so value estimates do not transfer reliably
  from random synthetic tensors to the matrix-multiplication tensor.
- The transformer learned synthetic decompositions, but additional target-aware
  learning through replay/self-play is needed.

## Next milestone

Target replay and fine-tuning have now been completed for three cycles. The best
seven-action trajectory improved to a residual with one nonzero entry, which can
be completed by one additional rank-one action and therefore yields an exact
eight-multiplication decomposition. It is not yet an exact rank-7 result.

The next improvement is symmetry-augmented target training with search-derived
multi-action policy targets, followed by another wide search.
