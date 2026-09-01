import numpy as np

from alphatensor.decomposition import (
    compose_decomposition,
    decomposition_residual,
    verify_decomposition,
)
from alphatensor.tensor import matrix_multiplication_tensor


# Factors are ordered as flattened A, flattened B, and flattened C.
# Over GF(2), subtraction and addition are both XOR.
STRASSEN_GF2 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
    ([1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]),
    ([0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
    ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def standard_eight_factor_decomposition():
    factors = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = np.zeros(4, dtype=np.uint8)
                v = np.zeros(4, dtype=np.uint8)
                w = np.zeros(4, dtype=np.uint8)
                u[i * 2 + j] = 1
                v[j * 2 + k] = 1
                w[i * 2 + k] = 1
                factors.append((u, v, w))
    return factors


def test_standard_eight_factor_decomposition_is_exact() -> None:
    target = matrix_multiplication_tensor(2)
    factors = standard_eight_factor_decomposition()

    assert len(factors) == 8
    assert verify_decomposition(target, factors)
    np.testing.assert_array_equal(compose_decomposition(factors), target)


def test_strassen_seven_factor_decomposition_is_exact_over_gf2() -> None:
    target = matrix_multiplication_tensor(2)

    assert len(STRASSEN_GF2) == 7
    assert verify_decomposition(target, STRASSEN_GF2)
    assert not np.any(decomposition_residual(target, STRASSEN_GF2))


def test_incomplete_decomposition_is_rejected() -> None:
    target = matrix_multiplication_tensor(2)
    incomplete = STRASSEN_GF2[:-1]
    residual = decomposition_residual(target, incomplete)

    assert not verify_decomposition(target, incomplete)
    assert np.any(residual)
