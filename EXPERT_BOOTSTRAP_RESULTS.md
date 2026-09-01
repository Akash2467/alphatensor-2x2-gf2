# Expert-Bootstrapped Rank-7 Policy Result

## Goal

Teach the transformer to reproduce an exact seven-multiplication decomposition,
then test it with the constraint solver completely disabled.

## Curriculum

- Source solution: `[24, 332, 1117, 1646, 2479, 2888, 3315]`
- Every nonterminal subset of the seven factors was converted into a residual
  state: `2^7 - 1 = 127` original states.
- Every factor still absent from a residual was assigned as a valid soft-policy
  target. This represents every valid ordering of the seven commuting actions.
- Four rank-preserving basis transforms expanded the replay to 635 unique states.
- The existing MCTS self-play transformer was fine-tuned, retaining a small
  synthetic-data pass each epoch to reduce catastrophic forgetting.

## Result

- Success epoch: 6 of 8 scheduled (early stop on exact rank 7)
- Expert-state top-5 accuracy: 97.480%
- Generic synthetic validation top-5: 21.278%
- Saved checkpoint: `checkpoints/expert_bootstrap_transformer.pt`
- Saved replay: `replay/expert_policy_replay.npz`
- Full history: `results/expert_bootstrap_history.json`

Policy-guided beam search returned:

`[24, 2479, 1117, 2888, 1646, 3315, 332]`

This is the same factor set as the source solution in a different valid order.
The residual was exactly zero after action 7.

## Independence of evaluation

- SAT/constraint solver during evaluation: **disabled**
- Exact one/two-action tail lookup: **disabled**
- Beam width: 256
- Policy actions considered per state: 64
- Expanded beam nodes: 83,110
- Exact tensor reconstruction: passed
- Exhaustive 2x2 GF(2) matrix tests: 256/256 passed

The separate 1,000-simulation MCTS evaluation did not find rank 7; it is the
transformer-guided beam search that produced the successful path.

## Interpretation

The reinforcement-learning policy can now produce a rank-7 result after being
bootstrapped with an exact solver solution. This is successful policy learning,
but it is not an independent RL discovery: the solver supplied the expert labels.
