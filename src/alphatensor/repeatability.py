"""Aggregate seeded solver and search repeatability measurements."""

from __future__ import annotations

from collections.abc import Sequence


def summarize_repeatability(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    solver_solutions = [
        tuple(int(index) for index in run["solver_action_indices"])  # type: ignore[index]
        for run in runs
        if run.get("solver_status") == "sat"
    ]
    return {
        "seeds_run": len(runs),
        "solver_sat_count": len(solver_solutions),
        "unique_solver_solutions": len(set(solver_solutions)),
        "all_solver_results_exact": bool(runs)
        and all(bool(run.get("solver_exact")) for run in runs),
        "all_exhaustive_checks_passed": bool(runs)
        and all(bool(run.get("exhaustive_exact")) for run in runs),
        "mcts_rank7_count": sum(bool(run.get("mcts_success")) for run in runs),
        "best_mcts_residual_rank": min(
            (int(run["mcts_residual_rank"]) for run in runs),
            default=None,
        ),
    }
