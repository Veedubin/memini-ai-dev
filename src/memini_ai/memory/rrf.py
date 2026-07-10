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
from typing import Any, TypeVar

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


# =============================================================================
# Multi-model RRF search helper (v0.12.0+)
# =============================================================================

# Column name → model name mapping for multi-model dispatch
# Use the FULL HuggingFace model names so ModelManager.embed(model_name=...) works
COLUMN_TO_MODEL: dict[str, str] = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_bge_m3": "BAAI/bge-m3",
    "embedding_bge_large": "BAAI/bge-large-en-v1.5",
}

# Model name → dimension mapping (mirrors model/manager.py MODEL_DIMS)
MODEL_TO_DIM: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
}


async def rrf_search(
    conn: Any,
    query_text: str,
    embedder: Any,
    k: int = 60,
    top_k_per_model: int = 20,
    final_top_k: int = 10,
    enabled_columns: dict[str, int] | None = None,
    distance_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Run RRF search across all enabled model vector spaces.

    For each enabled model, embeds the query and runs top-k vector search
    in that model's column, then merges the ranked lists using RRF.

    Args:
        conn: asyncpg Connection (already initialized with pgvector codec).
        query_text: Query string to search for.
        embedder: Object with an async ``embed(text, model_name)`` method
            that returns ``(list[float], model_name)``.
        k: RRF constant (default 60).
        top_k_per_model: Results to fetch from each model (default 20).
        final_top_k: Results to return after fusion (default 10).
        enabled_columns: Dict mapping column name to dimension.
            If None, uses all three: embedding(384), embedding_bge_m3(1024),
            embedding_bge_large(1024).
        distance_threshold: Max cosine distance for filtering (default 1.0 = all).

    Returns:
        List of dicts with keys: id, text, trust_score, embedding_model,
        rrf_score, ranks, best_distance.
    """
    if enabled_columns is None:
        enabled_columns = {
            "embedding": 384,
            "embedding_bge_m3": 1024,
            "embedding_bge_large": 1024,
        }

    # Build SQL per column
    column_sql: dict[str, str] = {
        "embedding": """
            SELECT id, text, trust_score, embedding_model,
                   embedding <=> $1::vector as distance
            FROM memories
            WHERE embedding IS NOT NULL
              AND embedding <=> $1::vector < $2
              AND is_archived = false
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """,
        "embedding_bge_m3": """
            SELECT id, text, trust_score, embedding_model,
                   embedding_bge_m3 <=> $1::vector as distance
            FROM memories
            WHERE embedding_bge_m3 IS NOT NULL
              AND embedding_bge_m3 <=> $1::vector < $2
              AND is_archived = false
            ORDER BY embedding_bge_m3 <=> $1::vector
            LIMIT $3
        """,
        "embedding_bge_large": """
            SELECT id, text, trust_score, embedding_model,
                   embedding_bge_large <=> $1::vector as distance
            FROM memories
            WHERE embedding_bge_large IS NOT NULL
              AND embedding_bge_large <=> $1::vector < $2
              AND is_archived = false
            ORDER BY embedding_bge_large <=> $1::vector
            LIMIT $3
        """,
    }

    all_results: dict[str, dict[str, Any]] = {}

    for column, _dim in enabled_columns.items():
        model_name = COLUMN_TO_MODEL.get(column)
        if model_name is None:
            continue
        sql = column_sql.get(column)
        if sql is None:
            continue

        # Embed the query with this model
        try:
            query_vector, used_model = await embedder.embed(
                query_text, model_name=model_name
            )
        except Exception:
            # Model not available — skip this column
            continue

        # Run top-k search
        try:
            rows = await conn.fetch(
                sql, query_vector, distance_threshold, top_k_per_model
            )
        except Exception:
            # Column might not exist — skip
            continue

        # Update RRF scores
        for rank, row in enumerate(rows, start=1):
            mid = str(row["id"])
            rrf_contribution = 1.0 / (k + rank)
            if mid not in all_results:
                all_results[mid] = {
                    "id": mid,
                    "text": row["text"],
                    "trust_score": (
                        float(row["trust_score"])
                        if row["trust_score"] is not None
                        else 0.5
                    ),
                    "embedding_model": row["embedding_model"],
                    "rrf_score": 0.0,
                    "ranks": {},
                    "best_distance": float(row["distance"]),
                }
            all_results[mid]["rrf_score"] += rrf_contribution
            all_results[mid]["ranks"][model_name] = rank
            dist = float(row["distance"])
            if dist < all_results[mid]["best_distance"]:
                all_results[mid]["best_distance"] = dist

    # Sort by RRF score, return top-k
    ranked = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:final_top_k]
