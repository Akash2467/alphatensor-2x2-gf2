from alphatensor.repeatability import summarize_repeatability


def test_repeatability_summary_counts_unique_solutions() -> None:
    runs = [
        {
            "solver_status": "sat",
            "solver_action_indices": [1, 2, 3],
            "solver_exact": True,
            "exhaustive_exact": True,
            "mcts_success": False,
            "mcts_residual_rank": 1,
        },
        {
            "solver_status": "sat",
            "solver_action_indices": [4, 5, 6],
            "solver_exact": True,
            "exhaustive_exact": True,
            "mcts_success": True,
            "mcts_residual_rank": 0,
        },
    ]
    summary = summarize_repeatability(runs)
    assert summary["solver_sat_count"] == 2
    assert summary["unique_solver_solutions"] == 2
    assert summary["all_exhaustive_checks_passed"]
    assert summary["mcts_rank7_count"] == 1
    assert summary["best_mcts_residual_rank"] == 0
