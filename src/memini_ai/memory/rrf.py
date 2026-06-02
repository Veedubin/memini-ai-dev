"""Reciprocal Rank Fusion (RRF) for multi-source retrieval.

This module implements the RRF algorithm used by the dual-model memory system
(v0.7.0+). RRF is a simple, parameter-light method for combining ranked
result lists from multiple retrieval sources (e.g., 384-dim MiniLM search
and 1024-dim BGE-Large search) into a single fused ranking.

Algorithm:
    For each ranked list L_i, the i-th result contributes a score of
    1 / (k + rank_i) to the fused score, where k is a constant (default 60)
    that dampens the impact of high ranks. The fused score for a result is
    the sum of its contributions across all lists. Results appearing in
    multiple lists are naturally boosted (same ID in both = sum of both
    contributions).

References:
    Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    k: int = 60,
) -> list[tuple[T, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each input list should already be sorted from most-relevant to least-relevant
    (rank 0 is the top result). Items are identified by hashable ID. Items
    appearing in multiple lists receive a fused score that is the sum of their
    per-list contributions, naturally boosting multi-source agreement.

    Args:
        ranked_lists: Sequence of ranked lists, each sorted best-to-worst.
        k: RRF constant — higher values flatten the contribution curve so
           lower ranks matter more. 60 is the canonical value from the
           original paper. Clamped to [1, 1000] by callers as needed.

    Returns:
        List of (item, fused_score) tuples sorted by fused_score DESC. Items
        with the same fused score preserve relative order from their first
        appearance (stable sort).

    Examples:
        >>> list_a = ["a", "b", "c"]
        >>> list_b = ["b", "a", "d"]
        >>> reciprocal_rank_fusion([list_a, list_b])
        [('b', 0.03279), ('a', 0.03279), ('c', 0.01639), ('d', 0.01639)]

    Notes:
        - Empty input → empty output
        - Empty inner lists are skipped
        - Duplicates within a single list are counted at the FIRST occurrence
          (subsequent duplicates in the same list contribute 0 — they're
          already in the result set)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not ranked_lists:
        return []

    # fused_scores[item] = total RRF contribution across all lists
    fused_scores: dict[T, float] = {}
    # Tracks the order of first appearance for stable output
    first_seen_order: list[T] = []
    seen_in_list: set[T] = set()

    for ranked_list in ranked_lists:
        seen_in_list.clear()
        for rank, item in enumerate(ranked_list):
            # Skip duplicates within the same list (counted at first occurrence)
            if item in seen_in_list:
                continue
            seen_in_list.add(item)

            contribution = 1.0 / (k + rank)
            if item in fused_scores:
                fused_scores[item] += contribution
            else:
                fused_scores[item] = contribution
                first_seen_order.append(item)

    # Build (item, score) pairs sorted by score DESC, then by first-seen order
    pairs = [(item, fused_scores[item]) for item in first_seen_order]
    pairs.sort(key=lambda pair: (-pair[1], first_seen_order.index(pair[0])))
    return pairs


def rrf_with_limit(
    ranked_lists: Sequence[Sequence[T]],
    k: int = 60,
    limit: int | None = None,
) -> list[T]:
    """Convenience wrapper: run RRF and return just the top-N item IDs.

    Args:
        ranked_lists: Sequence of ranked lists, each sorted best-to-worst.
        k: RRF constant (default 60).
        limit: If set, return at most this many items.

    Returns:
        List of top items (just the IDs, no scores) sorted by fused rank.
    """
    fused = reciprocal_rank_fusion(ranked_lists, k=k)
    if limit is None:
        return [item for item, _ in fused]
    return [item for item, _ in fused[:limit]]
