"""Exact subset-replacement search for shortening GF(2) decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .actions import GF2ActionSpace
from .decomposition import verify_decomposition


PAIR_INDEX_BITS = 12
PAIR_INDEX_MASK = (1 << PAIR_INDEX_BITS) - 1


@dataclass(frozen=True)
class LocalRepairResult:
    success: bool
    original_actions: tuple[int, ...]
    repaired_actions: tuple[int, ...]
    removed_positions: tuple[int, ...]
    removed_actions: tuple[int, ...]
    replacement_actions: tuple[int, ...]
    tested_subsets: int


def _tensor_codes(tensors: NDArray[np.uint8]) -> NDArray[np.uint64]:
    flat = np.ascontiguousarray(tensors.reshape(-1, 64), dtype=np.uint8)
    packed = np.packbits(flat, axis=1, bitorder="little")
    return packed.view("<u8").reshape(-1)


def _single_term_lookup(term_codes: NDArray[np.uint64]) -> dict[int, int]:
    return {int(code): index for index, code in enumerate(term_codes)}


def _build_pair_index(
    term_codes: NDArray[np.uint64],
) -> tuple[NDArray[np.uint64], NDArray[np.uint32]]:
    """Build a sorted compact index of all distinct unordered action pairs."""
    action_count = int(term_codes.size)
    pair_count = action_count * (action_count - 1) // 2
    pair_codes = np.empty(pair_count, dtype=np.uint64)
    pair_actions = np.empty(pair_count, dtype=np.uint32)
    offset = 0
    for left in range(action_count - 1):
        count = action_count - left - 1
        end = offset + count
        right = np.arange(left + 1, action_count, dtype=np.uint32)
        pair_codes[offset:end] = term_codes[left] ^ term_codes[left + 1 :]
        pair_actions[offset:end] = (np.uint32(left) << PAIR_INDEX_BITS) | right
        offset = end
    order = np.argsort(pair_codes, kind="stable")
    return pair_codes[order], pair_actions[order]


def _valid_replacement(
    actions: tuple[int, ...], forbidden: set[int]
) -> bool:
    return len(set(actions)) == len(actions) and not (set(actions) & forbidden)


def _find_replacement(
    residual_code: int,
    replacement_count: int,
    *,
    term_codes: NDArray[np.uint64],
    term_lookup: dict[int, int],
    forbidden: set[int],
    pair_index: tuple[NDArray[np.uint64], NDArray[np.uint32]] | None,
) -> tuple[int, ...] | None:
    if replacement_count == 1:
        action = term_lookup.get(residual_code)
        if action is not None and action not in forbidden:
            return (action,)
        return None

    if replacement_count == 2:
        for first, first_code in enumerate(term_codes):
            if first in forbidden:
                continue
            second = term_lookup.get(residual_code ^ int(first_code))
            candidate = (first, second) if second is not None else ()
            if candidate and _valid_replacement(candidate, forbidden):
                return candidate
        return None

    if replacement_count == 3:
        if pair_index is None:
            raise ValueError("a pair index is required for three-action repair")
        pair_codes, pair_actions = pair_index
        for first, first_code in enumerate(term_codes):
            if first in forbidden:
                continue
            wanted = np.uint64(residual_code ^ int(first_code))
            lower = int(np.searchsorted(pair_codes, wanted, side="left"))
            upper = int(np.searchsorted(pair_codes, wanted, side="right"))
            for packed in pair_actions[lower:upper]:
                packed_int = int(packed)
                left = packed_int >> PAIR_INDEX_BITS
                right = packed_int & PAIR_INDEX_MASK
                candidate = (first, left, right)
                if _valid_replacement(candidate, forbidden):
                    return candidate
        return None

    if replacement_count == 4:
        if pair_index is None:
            raise ValueError("a pair index is required for four-action repair")
        pair_codes, pair_actions = pair_index
        chunk_size = 250_000
        for start in range(0, len(pair_codes), chunk_size):
            end = min(start + chunk_size, len(pair_codes))
            first_codes = pair_codes[start:end]
            wanted = np.bitwise_xor(first_codes, np.uint64(residual_code))
            lower = np.searchsorted(pair_codes, wanted, side="left")
            in_range = lower < len(pair_codes)
            matched = np.zeros(len(first_codes), dtype=bool)
            matched[in_range] = pair_codes[lower[in_range]] == wanted[in_range]
            for local_index in np.flatnonzero(matched):
                first_packed = int(pair_actions[start + int(local_index)])
                first_left = first_packed >> PAIR_INDEX_BITS
                first_right = first_packed & PAIR_INDEX_MASK
                first_pair = (first_left, first_right)
                if not _valid_replacement(first_pair, forbidden):
                    continue
                code = wanted[local_index]
                range_start = int(lower[local_index])
                range_end = int(np.searchsorted(pair_codes, code, side="right"))
                for packed in pair_actions[range_start:range_end]:
                    packed_int = int(packed)
                    second_left = packed_int >> PAIR_INDEX_BITS
                    second_right = packed_int & PAIR_INDEX_MASK
                    candidate = first_pair + (second_left, second_right)
                    if _valid_replacement(candidate, forbidden):
                        return candidate
        return None

    raise ValueError("replacement_count must be between 1 and 4")


def local_repair_decomposition(
    target: ArrayLike,
    action_indices: tuple[int, ...] | list[int],
    *,
    min_removed: int = 2,
    max_removed: int = 4,
    action_space: GF2ActionSpace | None = None,
) -> LocalRepairResult:
    """Try replacing 2..5 selected actions with one fewer exact actions.

    The input decomposition must already be exact. Every proposed repair is
    independently verified before a successful result is returned.
    """
    if not 2 <= min_removed <= max_removed <= 5:
        raise ValueError("removed-action bounds must satisfy 2 <= min <= max <= 5")
    catalog = action_space or GF2ActionSpace()
    original = tuple(int(index) for index in action_indices)
    if len(original) < 2:
        raise ValueError("at least two actions are required")
    factors = [catalog.action(index) for index in original]
    if not verify_decomposition(target, factors):
        raise ValueError("action_indices are not an exact decomposition of target")

    term_codes = _tensor_codes(catalog.terms)
    term_lookup = _single_term_lookup(term_codes)
    pair_index: tuple[NDArray[np.uint64], NDArray[np.uint32]] | None = None
    tested = 0

    for removed_count in range(min_removed, min(max_removed, len(original)) + 1):
        replacement_count = removed_count - 1
        if replacement_count >= 3 and pair_index is None:
            pair_index = _build_pair_index(term_codes)
        for positions in combinations(range(len(original)), removed_count):
            tested += 1
            removed = tuple(original[position] for position in positions)
            residual_code = 0
            for action in removed:
                residual_code ^= int(term_codes[action])
            position_set = set(positions)
            kept = tuple(
                action for position, action in enumerate(original) if position not in position_set
            )
            replacement = _find_replacement(
                residual_code,
                replacement_count,
                term_codes=term_codes,
                term_lookup=term_lookup,
                forbidden=set(kept),
                pair_index=pair_index,
            )
            if replacement is None:
                continue
            repaired = kept + replacement
            if verify_decomposition(
                target, [catalog.action(index) for index in repaired]
            ):
                return LocalRepairResult(
                    success=True,
                    original_actions=original,
                    repaired_actions=repaired,
                    removed_positions=positions,
                    removed_actions=removed,
                    replacement_actions=replacement,
                    tested_subsets=tested,
                )

    return LocalRepairResult(
        success=False,
        original_actions=original,
        repaired_actions=original,
        removed_positions=(),
        removed_actions=(),
        replacement_actions=(),
        tested_subsets=tested,
    )
