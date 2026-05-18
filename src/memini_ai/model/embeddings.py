"""Embedding generation with batching support."""

from __future__ import annotations

import asyncio
import time
from math import ceil

from pydantic import BaseModel

from memini_ai.model.manager import ModelManager


class EmbeddingResult(BaseModel):
    """Result of an embedding generation operation."""

    embedding: list[float]
    token_count: int
    model_id: str
    device: str
    timestamp: int
    latency_ms: int


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Approximation: ceil(len(text) / 4) for typical English text.

    Args:
        text: Input text string.

    Returns:
        Estimated number of tokens.
    """
    return ceil(len(text) / 4)


async def generate_embedding(text: str) -> EmbeddingResult:
    """Generate a single embedding from text.

    Args:
        text: Input text to embed.

    Returns:
        EmbeddingResult with embedding vector and metadata.
    """
    results = await generate_embeddings([text], batch_size=1)
    return results[0]


async def generate_embeddings(
    texts: list[str],
    batch_size: int = 8,
) -> list[EmbeddingResult]:
    """Generate embeddings for multiple texts with batching.

    Args:
        texts: List of input texts to embed.
        batch_size: Number of texts to process in each batch.

    Returns:
        List of EmbeddingResult objects, one per input text.
    """
    if not texts:
        return []

    manager = ModelManager.get_instance()
    model = await manager.acquire()

    metadata = manager.get_metadata()
    results: list[EmbeddingResult] = []

    try:
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Encode batch using asyncio.to_thread for async operation
            start_time = time.perf_counter()
            vectors = await asyncio.to_thread(model.encode, batch)
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            # Convert to EmbeddingResult objects
            for j, vector in enumerate(vectors):
                text = batch[j]
                embedding_list = (
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )

                results.append(
                    EmbeddingResult(
                        embedding=embedding_list,
                        token_count=_estimate_tokens(text),
                        model_id=metadata.model_id,
                        device=metadata.device,
                        timestamp=int(time.time() * 1000),
                        latency_ms=latency_ms,
                    )
                )

    finally:
        manager.release()

    return results
