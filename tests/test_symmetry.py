import numpy as np

from alphatensor.actions import GF2ActionSpace
from alphatensor.decomposition import verify_decomposition
from alphatensor.symmetry import (
    gf2_inverse,
    random_basis_transform,
    random_invertible_matrix,
    transform_action_index,
    transform_tensor,
)
from alphatensor.tensor import matrix_multiplication_tensor


def test_random_gf2_matrix_has_exact_inverse() -> None:
    rng = np.random.default_rng(5)
    matrix = random_invertible_matrix(rng)
    inverse = gf2_inverse(matrix)
    np.testing.assert_array_equal(
        np.remainder(matrix @ inverse, 2), np.eye(4, dtype=np.uint8)
    )


def test_action_transform_matches_transformed_rank_one_tensor() -> None:
    rng = np.random.default_rng(7)
    transform = random_basis_transform(rng)
    action_space = GF2ActionSpace()
    action_index = 1234
    transformed_index = transform_action_index(
        action_index, transform, action_space=action_space
    )
    np.testing.assert_array_equal(
        transform_tensor(action_space.term(action_index), transform),
        action_space.term(transformed_index),
    )


def test_symmetry_preserves_exact_decomposition() -> None:
    rng = np.random.default_rng(11)
    transform = random_basis_transform(rng)
    action_space = GF2ActionSpace()
    target = matrix_multiplication_tensor(2)
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    action_indices = [action_space.index(factor) for factor in strassen_factors]
    transformed_target = transform_tensor(target, transform)
    transformed_factors = [
        action_space.action(
            transform_action_index(index, transform, action_space=action_space)
        )
        for index in action_indices
    ]
    assert verify_decomposition(transformed_target, transformed_factors)
