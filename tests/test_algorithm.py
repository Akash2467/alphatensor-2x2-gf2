from alphatensor.actions import GF2ActionSpace
from alphatensor.algorithm import describe_equations, exhaustive_verify_2x2


SOLVED_RANK7 = [24, 332, 1117, 1646, 2479, 2888, 3315]


def test_solved_rank7_algorithm_passes_all_binary_inputs() -> None:
    action_space = GF2ActionSpace()
    factors = [action_space.action(index) for index in SOLVED_RANK7]
    exact, checked = exhaustive_verify_2x2(factors)
    assert exact
    assert checked == 256


def test_rank7_equation_export_has_products_and_outputs() -> None:
    action_space = GF2ActionSpace()
    factors = [action_space.action(index) for index in SOLVED_RANK7]
    equations = describe_equations(factors)
    assert len(equations) == 11
    assert equations[0].startswith("p1 =")
    assert equations[-1].startswith("c22 =")
