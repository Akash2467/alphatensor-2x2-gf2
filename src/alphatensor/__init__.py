"""Core mathematical utilities for the AlphaTensor prototype."""

from .actions import (
    ACTION_COUNT,
    NONZERO_VECTOR_COUNT,
    GF2ActionSpace,
    code_from_vector,
    vector_from_code,
)
from .decomposition import (
    compose_decomposition,
    decomposition_residual,
    verify_decomposition,
)
from .game import StepResult, TensorGame, gf2_matrix_rank, slice_rank_upper_bound
from .synthetic import (
    SyntheticDataset,
    SyntheticGame,
    generate_synthetic_dataset,
    generate_synthetic_game,
)
from .tensor import matrix_multiplication_tensor, rank_one_tensor

__all__ = [
    "ACTION_COUNT",
    "code_from_vector",
    "compose_decomposition",
    "decomposition_residual",
    "GF2ActionSpace",
    "gf2_matrix_rank",
    "generate_synthetic_dataset",
    "generate_synthetic_game",
    "matrix_multiplication_tensor",
    "NONZERO_VECTOR_COUNT",
    "rank_one_tensor",
    "slice_rank_upper_bound",
    "StepResult",
    "SyntheticDataset",
    "SyntheticGame",
    "TensorGame",
    "vector_from_code",
    "verify_decomposition",
]
