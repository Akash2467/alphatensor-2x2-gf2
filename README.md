# AlphaTensor-Style 2x2 Matrix Multiplication

This project is a compact AlphaTensor-style research prototype for exact 2x2
matrix multiplication over GF(2). It includes a TensorGame environment, all
3,375 legal rank-one actions, transformer policy/value learning, supervised and
self-play training, beam search, MCTS, symmetry augmentation, replay systems,
local exact repair, a global SAT-based rank-7 solver, and solver-bootstrapped
expert fine-tuning.

## Final result

The project independently discovered exact seven-multiplication decompositions.
The primary solution is:

`[24, 332, 1117, 1646, 2479, 2888, 3315]`

It passes exact tensor reconstruction and all 256 binary 2x2 matrix input pairs.
Three solver seeds produced three distinct exact rank-7 solutions. After expert
bootstrapping on all valid subset states from one solution family, the learned
transformer produced a complete rank-7 path with policy-guided beam search. The
evaluation used neither the SAT solver nor exact-tail completion.

## Setup

From PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

Run all tests:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

Expected final result: `60 passed`.

## Important commands

Run global rank-7 discovery:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\solve_rank7.py --rank 7 --timeout-seconds 300
```

Exhaustively verify the discovered algorithm:

```powershell
.\.venv\Scripts\python.exe scripts\verify_rank7_solution.py
```

Repeat exact solving and stochastic MCTS across seeds:

```powershell
.\.venv\Scripts\python.exe scripts\run_repeatability.py --seeds 0 1 2
```

Run the best learned policy with MCTS:

```powershell
.\.venv\Scripts\python.exe scripts\run_mcts.py `
  --checkpoint checkpoints\mcts_self_play_transformer.pt `
  --simulations 10000 --top-k 256
```

Reproduce solver-bootstrapped fine-tuning and solver-free evaluation:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\expert_bootstrap_finetune.py `
  --solution-limit 1 --epochs 8 --expert-passes 8 `
  --symmetry-augmentations 4 --synthetic-games 200 `
  --learning-rate 0.0001 --beam-width 256 --beam-top-k 64
```

## Main components

- `src/alphatensor/tensor.py`: multiplication and rank-one tensors
- `src/alphatensor/game.py`: TensorGame transitions and rewards
- `src/alphatensor/actions.py`: complete finite action catalog
- `src/alphatensor/model.py`: transformer/MLP and training losses
- `src/alphatensor/search.py`: policy-guided beam search
- `src/alphatensor/mcts.py`: PUCT and batched multi-tree MCTS
- `src/alphatensor/replay.py`: hard multi-action target replay
- `src/alphatensor/policy_replay.py`: sparse visit-distribution replay
- `src/alphatensor/symmetry.py`: rank-preserving GF(2) augmentation
- `src/alphatensor/repair.py`: exact local meet-in-the-middle repair
- `src/alphatensor/constraint.py`: global exact rank constraints
- `src/alphatensor/expert.py`: exact-solution subset policy targets
- `src/alphatensor/algorithm.py`: algorithm execution and exhaustive checking

## Documentation

- `PROJECT_FLOW.md`: complete development flow
- `RANK7_ALGORITHM.md`: readable seven-multiplication equations
- `RANK7_SOLVER_RESULTS.md`: exact solver and policy comparison
- `REPEATABILITY_RESULTS.md`: multi-seed evaluation
- `EXPERT_BOOTSTRAP_RESULTS.md`: learned-policy rank-7 result
- `ROBUSTNESS_RESULTS.md`: beam-width, noisy-seed and generalization evaluation
- Other `*_RESULTS.md` files record each training and search milestone.

## Scope and limitation

This is intentionally restricted to 2x2 matrices over GF(2). It reproduces the
core AlphaTensor workflow at small scale, not DeepMind's production-scale agent.
The global solver supplies the expert curriculum. After that bootstrapping, the
learned policy can reproduce rank 7 through beam search, but this is not an
independent rediscovery because the training labels came from a solver solution.
The robustness study found a minimum successful beam width of 256 and 5/5
successes under small policy perturbations; every success used the taught factor
set, so novel-solution generalization has not yet been demonstrated.
