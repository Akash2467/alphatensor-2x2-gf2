from alphatensor.actions import GF2ActionSpace
from alphatensor.constraint import solve_exact_rank
from alphatensor.tensor import matrix_multiplication_tensor


def test_global_constraint_solver_recovers_rank_one_term() -> None:
    action_space = GF2ActionSpace()
    target = action_space.term(731)
    result = solve_exact_rank(
        target, 1, action_space=action_space, timeout_seconds=10
    )
    assert result.status == "sat"
    assert result.action_indices == (731,)
    assert result.exact_verification


def test_constraint_solver_rejects_invalid_target_shape() -> None:
    try:
        solve_exact_rank(matrix_multiplication_tensor(1), 1)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid target shape was accepted")
