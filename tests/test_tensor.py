import numpy as np
import pytest

from alphatensor.tensor import matrix_multiplication_tensor, rank_one_tensor


def test_2x2_matrix_multiplication_tensor_shape_and_entries() -> None:
    tensor = matrix_multiplication_tensor(2)

    assert tensor.shape == (4, 4, 4)
    assert tensor.dtype == np.uint8
    assert int(tensor.sum()) == 8


def test_tensor_encodes_matrix_multiplication_over_gf2() -> None:
    tensor = matrix_multiplication_tensor(2)
    a = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    b = np.array([[0, 1], [1, 1]], dtype=np.uint8)

    encoded_product = np.einsum("abc,a,b->c", tensor, a.ravel(), b.ravel()) % 2
    expected_product = (a @ b) % 2

    np.testing.assert_array_equal(encoded_product.reshape(2, 2), expected_product)


def test_rank_one_tensor() -> None:
    u = np.array([1, 0, 1, 0], dtype=np.uint8)
    v = np.array([0, 1, 0, 0], dtype=np.uint8)
    w = np.array([1, 1, 0, 0], dtype=np.uint8)

    result = rank_one_tensor(u, v, w)

    assert result.shape == (4, 4, 4)
    assert int(result.sum()) == 4
    assert result[0, 1, 0] == 1
    assert result[2, 1, 1] == 1


def test_rejects_non_binary_coefficients() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        rank_one_tensor([0, 2], [1, 0], [1, 1])
