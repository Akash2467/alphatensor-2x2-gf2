from scripts.evaluate_expert_robustness import classify_solution


def test_classify_solution_ignores_action_order() -> None:
    known = ((1, 2, 3), (4, 5, 6))
    assert classify_solution((3, 1, 2), known) == "known_solution_1"
    assert classify_solution((6, 4, 5), known) == "known_solution_2"


def test_classify_solution_marks_unseen_complete_set_as_novel() -> None:
    known = ((1, 2, 3, 4, 5, 6, 7),)
    assert classify_solution((8, 9, 10, 11, 12, 13, 14), known) == "novel"
    assert classify_solution((8, 9), known) == "incomplete"
