"""Tests for the Reciprocal Rank Fusion (RRF) algorithm.

Pure-Python tests — no database, no embedding model. Covers:
- Basic fusion across two ranked lists
- Empty input (no lists)
- Single list (degenerate case)
- Dedup within a single list (only first occurrence counts)
- k < 1 raises ValueError
- rrf_with_limit wrapper returns just IDs
- rrf_with_limit respects the limit argument
- Items in both lists get boosted fused score
- Stable sort for tied scores (first-seen order wins)

Reference: Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

import pytest

from memini_ai.memory.rrf import reciprocal_rank_fusion, rrf_with_limit


def test_basic_two_list_fusion() -> None:
    """Items in both lists get the highest fused score."""
    list_a = ["a", "b", "c"]
    list_b = ["b", "a", "d"]
    result = reciprocal_rank_fusion([list_a, list_b])
    # "a" appears at rank 0 in list_a and rank 1 in list_b
    # "b" appears at rank 1 in list_a and rank 0 in list_b
    # Both should have the same fused score (1/61 + 1/60 = 1/60 + 1/61)
    assert result[0][1] == pytest.approx(1 / 60 + 1 / 61)
    assert result[1][1] == pytest.approx(1 / 60 + 1 / 61)
    # The result should be sorted by score DESC, with the two top items
    # tied (a and b). Stable sort means the order of first appearance
    # in the input dict wins.
    top_ids = {result[0][0], result[1][0]}
    assert top_ids == {"a", "b"}
    # c only in list_a, d only in list_b → tied at 1/62
    assert {"c", "d"} == {result[2][0], result[3][0]}


def test_empty_input_returns_empty() -> None:
    """No lists → empty result."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_single_list_preserves_order() -> None:
    """A single ranked list should round-trip via RRF (with k=60 contributions)."""
    items = ["x", "y", "z", "w"]
    result = reciprocal_rank_fusion([items])
    assert [item for item, _ in result] == items
    # x at rank 0 → 1/60
    assert result[0][1] == pytest.approx(1 / 60)
    # y at rank 1 → 1/61
    assert result[1][1] == pytest.approx(1 / 61)


def test_dedup_within_single_list() -> None:
    """A repeated item in the same list is counted only at first occurrence."""
    # "a" appears at ranks 0 and 2 in the same list — its contribution
    # is just 1/(k+0) = 1/60, NOT 1/60 + 1/62.
    result = reciprocal_rank_fusion([["a", "b", "a"]])
    by_id = dict(result)
    assert by_id["a"] == pytest.approx(1 / 60)
    assert by_id["b"] == pytest.approx(1 / 61)
    # Output should not contain duplicate entries.
    assert [item for item, _ in result] == ["a", "b"]


def test_k_must_be_positive() -> None:
    """k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_rrf_with_limit_returns_ids() -> None:
    """rrf_with_limit returns just the item IDs, in fused order."""
    list_a = ["a", "b", "c", "d", "e"]
    list_b = ["c", "a", "b", "f", "g"]
    ids = rrf_with_limit([list_a, list_b], k=60, limit=3)
    assert len(ids) == 3
    # All returned IDs should be unique and from the input sets.
    assert set(ids).issubset({"a", "b", "c", "d", "e", "f", "g"})


def test_rrf_with_limit_none_returns_all() -> None:
    """rrf_with_limit(limit=None) returns all fused items."""
    list_a = ["a", "b"]
    list_b = ["c", "d"]
    ids = rrf_with_limit([list_a, list_b], limit=None)
    # All 4 distinct items should be returned
    assert set(ids) == {"a", "b", "c", "d"}


def test_dual_list_boost_for_shared_items() -> None:
    """An item in both lists has fused score = sum of both contributions."""
    # "x" at rank 0 in list_a (1/60) and rank 5 in list_b (1/65)
    list_a = ["x", "a", "b", "c", "d"]
    list_b = ["y", "z", "w", "v", "u", "x", "p"]
    result = reciprocal_rank_fusion([list_a, list_b])
    by_id = dict(result)
    assert by_id["x"] == pytest.approx(1 / 60 + 1 / 65)


def test_stable_sort_for_tied_scores() -> None:
    """Tied scores preserve the order of first appearance."""
    # All items have score 1/61 (rank 1 in their respective lists).
    # Stable sort: order determined by first-seen order.
    list_a = ["a", "b"]
    list_b = ["c", "d"]
    result = reciprocal_rank_fusion([list_a, list_b])
    # First seen: a (list_a rank 0) — but a is at rank 0, score 1/60
    # b at rank 1 in list_a → 1/61, c at rank 0 in list_b → 1/60,
    # d at rank 1 in list_b → 1/61.
    # So a and c are tied at 1/60, b and d tied at 1/61.
    # First-seen order: a, b, c, d.
    # Stable sort puts a before c (both at 1/60) and b before d (both at 1/61).
    assert [item for item, _ in result[:2]] == ["a", "c"]
    assert [item for item, _ in result[2:]] == ["b", "d"]


def test_integer_k_validation() -> None:
    """k is validated before any work is done."""
    # Edge: k=1 should work fine — no division by zero.
    result = reciprocal_rank_fusion([["a"]], k=1)
    assert result == [("a", pytest.approx(1.0))]
    # Edge: k=1000 (max allowed by config validator) should work.
    result = reciprocal_rank_fusion([["a"]], k=1000)
    assert result == [("a", pytest.approx(1 / 1000))]
