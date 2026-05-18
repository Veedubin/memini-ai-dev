"""Model package - embedding model management."""

from memini_ai.model.embeddings import (
    EmbeddingResult,
    generate_embedding,
    generate_embeddings,
)
from memini_ai.model.manager import ModelManager

__all__ = [
    "ModelManager",
    "EmbeddingResult",
    "generate_embedding",
    "generate_embeddings",
]
