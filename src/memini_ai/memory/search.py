"""Memory search layer - 4 search strategies with BM25 and RRF fusion."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.schema import MemoryEntry, SearchOptions, SearchStrategy
from memini_ai.model.embeddings import generate_embedding

if TYPE_CHECKING:
    pass


class MemorySearch:
    """Memory search with 4 strategies: TIERED, VECTOR_ONLY, TEXT_ONLY, PARALLEL.

    Uses BM25 for text search and RRF for result fusion in parallel mode.
    """

    def __init__(self, db: VectorDatabase) -> None:
        """Initialize MemorySearch.

        Args:
            db: VectorDatabase instance for vector operations.
        """
        self._db = db
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[MemoryEntry] = []
        self._bm25_lock = asyncio.Lock()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Simple word splitting with lowercase.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()

    async def _build_bm25_index(self) -> None:
        """Build or rebuild BM25 index from all memories."""
        async with self._bm25_lock:
            # Get all memories
            memories = await self._db.list_memories()

            # Handle empty corpus - skip BM25 index building
            if not memories:
                self._bm25_index = None
                self._bm25_corpus = []
                return

            # Tokenize corpus
            corpus_tokens = [self._tokenize(m.text) for m in memories]

            # Build BM25 index
            self._bm25_index = BM25Okapi(corpus_tokens)
            self._bm25_corpus = memories

    async def _ensure_bm25(self) -> None:
        """Ensure BM25 index is built."""
        if self._bm25_index is None or self._bm25_corpus is None:
            await self._build_bm25_index()

    def _normalize_bm25_scores(
        self,
        scores: list[float],
    ) -> list[float]:
        """Normalize BM25 scores to [0, 1] range.

        Args:
            scores: Raw BM25 scores.

        Returns:
            Normalized scores.
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _rrf_fusion(
        self,
        rankings: list[list[MemoryEntry]],
        scores: list[list[float]],
        k: float = 60,
    ) -> list[tuple[MemoryEntry, float]]:
        """Reciprocal Rank Fusion for combining rankings.

        Args:
            rankings: List of ranked result lists.
            scores: Corresponding score lists.
            k: RRF constant (default 60).

        Returns:
            Combined ranked list with fusion scores.
        """
        # Map memory ID to combined score
        memory_scores: dict[str, float] = {}
        memory_entries: dict[str, MemoryEntry] = {}

        for ranking, score_list in zip(rankings, scores, strict=True):
            for rank, (entry, _score) in enumerate(
                zip(ranking, score_list, strict=True)
            ):
                memory_entries[entry.id] = entry
                if entry.id not in memory_scores:
                    memory_scores[entry.id] = 0.0
                memory_scores[entry.id] += 1 / (k + rank + 1)

        # Sort by combined score
        sorted_items = sorted(
            memory_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(memory_entries[mid], score) for mid, score in sorted_items]

    async def query(
        self,
        question: str,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Main query entry point.

        Args:
            question: Query string.
            options: Search options.

        Returns:
            List of matching MemoryEntry objects.
        """
        if options is None:
            options = SearchOptions()

        strategy = options.strategy or SearchStrategy.TIERED

        if strategy == SearchStrategy.TIERED:
            return await self.tiered_search(question, options)
        elif strategy == SearchStrategy.VECTOR_ONLY:
            return await self.vector_only_search(question, options)
        elif strategy == SearchStrategy.TEXT_ONLY:
            return await self.text_only_search(question, options)
        elif strategy == SearchStrategy.PARALLEL:
            return await self.parallel_search(question, options)
        else:
            return await self.tiered_search(question, options)

    async def tiered_search(
        self,
        question: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """Tiered search: vector first, then text fallback.

        Strategy:
        1. Generate embedding for query
        2. Run vector search with topK*2 limit
        3. If top score >= threshold (0.72): return vector results
        4. Else: run text search and merge results (interleave)

        Args:
            question: Query string.
            options: Search options.

        Returns:
            List of MemoryEntry objects.
        """
        # Generate embedding
        embedding_result = await generate_embedding(question)
        vector = embedding_result.embedding

        # Vector search with doubled topK
        search_options = SearchOptions(
            topK=options.top_k * 2,
            strategy=SearchStrategy.VECTOR_ONLY,
            threshold=options.threshold,
            filter=options.filter,
        )
        vector_results = await self._db.query_memories(vector, search_options)

        # Check threshold
        if vector_results and (vector_results[0].score or 0) >= options.threshold:
            return vector_results[: options.top_k]

        # Text fallback
        text_results = await self.text_only_search(question, options)

        # Merge and interleave results
        if not vector_results:
            return text_results[: options.top_k]

        if not text_results:
            return vector_results[: options.top_k]

        # Interleave: alternate from each list
        merged: list[MemoryEntry] = []
        v_idx = 0
        t_idx = 0
        while len(merged) < options.top_k and (
            v_idx < len(vector_results) or t_idx < len(text_results)
        ):
            if v_idx < len(vector_results):
                merged.append(vector_results[v_idx])
                v_idx += 1
            if t_idx < len(text_results) and len(merged) < options.top_k:
                merged.append(text_results[t_idx])
                t_idx += 1

        return merged

    async def vector_only_search(
        self,
        question: str,
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Pure vector similarity search.

        Args:
            question: Query string.
            options: Search options.
            collection_name: Optional collection override.

        Returns:
            List of MemoryEntry objects.
        """
        embedding_result = await generate_embedding(question)
        vector = embedding_result.embedding

        return await self._db.query_memories(vector, options, collection_name)

    async def text_only_search(
        self,
        question: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """BM25 text search.

        Args:
            question: Query string.
            options: Search options.

        Returns:
            List of MemoryEntry objects.
        """
        await self._ensure_bm25()

        if self._bm25_index is None or not self._bm25_corpus:
            return []

        # Tokenize query
        query_tokens = self._tokenize(question)

        # Get BM25 scores
        raw_scores = self._bm25_index.get_scores(query_tokens)

        # Normalize scores to [0, 1]
        normalized_scores = self._normalize_bm25_scores(raw_scores.tolist())

        # Pair entries with scores and sort
        scored_entries = sorted(
            zip(self._bm25_corpus, normalized_scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top results
        results = [
            entry.model_copy(update={"score": score})
            for entry, score in scored_entries[: options.top_k]
        ]

        # Apply filter if specified
        if options.filter:
            filtered: list[MemoryEntry] = []
            for entry in results:
                if (
                    options.filter.source_type
                    and entry.source_type != options.filter.source_type
                ):
                    continue
                if (
                    options.filter.session_id
                    and entry.session_id != options.filter.session_id
                ):
                    continue
                if (
                    options.filter.project_id
                    and entry.project_id != options.filter.project_id
                ):
                    continue
                filtered.append(entry)
            return filtered

        return results

    async def parallel_search(
        self,
        question: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """Parallel vector + text search with RRF fusion.

        Args:
            question: Query string.
            options: Search options.

        Returns:
            List of MemoryEntry objects.
        """
        # Run vector and text searches concurrently
        vector_search = self.vector_only_search(question, options)
        text_search = self.text_only_search(question, options)

        vector_results, text_results = await asyncio.gather(vector_search, text_search)

        # Get scores
        vector_scores = [e.score or 0.0 for e in vector_results]
        text_scores = [e.score or 0.0 for e in text_results]

        # RRF fusion
        fused = self._rrf_fusion(
            [vector_results, text_results], [vector_scores, text_scores]
        )

        # Apply filter and return topK
        results: list[MemoryEntry] = []
        for entry, fusion_score in fused:
            entry_with_score = entry.model_copy(update={"score": fusion_score})

            if options.filter:
                if (
                    options.filter.source_type
                    and entry_with_score.source_type != options.filter.source_type
                ):
                    continue
                if (
                    options.filter.session_id
                    and entry_with_score.session_id != options.filter.session_id
                ):
                    continue
                if (
                    options.filter.project_id
                    and entry_with_score.project_id != options.filter.project_id
                ):
                    continue

            results.append(entry_with_score)

            if len(results) >= options.top_k:
                break

        return results

    async def search_with_vector(
        self,
        vector: list[float],
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Search with pre-computed vector.

        Args:
            vector: Pre-computed embedding vector.
            options: Search options.

        Returns:
            List of MemoryEntry objects.
        """
        if options is None:
            options = SearchOptions()
        return await self._db.query_memories(vector, options)

    async def query_with_fallback_collection(
        self,
        question: str,
        fallback_collection: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """Query with fallback to different dimension collection.

        Args:
            question: Query string.
            fallback_collection: Fallback collection name.
            options: Search options.

        Returns:
            List of MemoryEntry objects.
        """
        # Try primary first
        results = await self.vector_only_search(question, options)

        # If no results, try fallback
        if not results:
            results = await self.vector_only_search(
                question, options, fallback_collection
            )

        return results

    async def text_search_collection(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Text search within a specific collection.

        Args:
            query: Search query.
            collection_name: Collection to search.
            limit: Maximum results.

        Returns:
            List of MemoryEntry objects.
        """
        # Get all entries from collection
        entries = await self._db.scroll_collection(collection_name, limit=1000)

        if not entries:
            return []

        # Build temporary BM25 index
        corpus_tokens = [self._tokenize(e.text) for e in entries]
        bm25 = BM25Okapi(corpus_tokens)

        # Score
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        normalized = self._normalize_bm25_scores(scores.tolist())

        # Sort and return top results
        scored_entries = sorted(
            zip(entries, normalized, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            entry.model_copy(update={"score": score})
            for entry, score in scored_entries[:limit]
        ]

    async def get_similar(
        self,
        memory_id: str,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Find memories similar to a given memory.

        Args:
            memory_id: ID of the reference memory.
            options: Search options.

        Returns:
            List of similar MemoryEntry objects.
        """
        if options is None:
            options = SearchOptions()

        # Get the reference memory
        reference = await self._db.get_memory(memory_id)
        if reference is None or reference.vector is None:
            return []

        # Vector search with same filter
        return await self._db.query_memories(reference.vector, options)

    async def invalidate_bm25(self) -> None:
        """Invalidate BM25 cache (call after add/delete)."""
        self._bm25_index = None
        self._bm25_corpus = []
