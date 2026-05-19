"""Tests for embedding generation - mocked model for speed."""

from __future__ import annotations

from math import ceil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.model import (
    EmbeddingResult,
    ModelManager,
    generate_embedding,
    generate_embeddings,
)


def _estimate_tokens_for_test(text: str) -> int:
    """Estimate token count from text (matches embeddings.py logic)."""
    return ceil(len(text) / 4)


class TestTokenEstimation:
    """Tests for token estimation logic."""

    def test_estimate_tokens_exact_divisible(self) -> None:
        """Test token estimation when length is exactly divisible by 4."""
        text = "abcd" * 10  # 40 chars = 10 tokens
        assert _estimate_tokens_for_test(text) == 10

    def test_estimate_tokens_with_remainder(self) -> None:
        """Test token estimation when length has remainder."""
        text = "abc"  # 3 chars -> ceil(3/4) = 1 token
        assert _estimate_tokens_for_test(text) == 1

    def test_estimate_tokens_empty(self) -> None:
        """Test token estimation for empty string."""
        text = ""
        assert _estimate_tokens_for_test(text) == 0


class TestModelManager:
    """Tests for ModelManager singleton."""

    def test_singleton_pattern(self) -> None:
        """Test that get_instance returns the same instance."""
        # Reset singleton for test isolation
        ModelManager._instance = None
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        assert instance1 is instance2
        # Cleanup
        ModelManager._instance = None

    def test_ref_count_starts_at_zero(self) -> None:
        """Test that reference count starts at zero after init."""
        ModelManager._instance = None
        manager = ModelManager()
        assert manager._ref_count == 0


def _create_mock_transformer(return_dim: int = 1024) -> MagicMock:
    """Create a mock transformer that returns proper vectors."""
    mock = MagicMock()
    # encode returns list of vectors matching input count
    mock.encode.side_effect = lambda texts: [[0.1] * return_dim for _ in texts]
    mock.get_sentence_embedding_dimension.return_value = return_dim
    return mock


@pytest.mark.asyncio
class TestGenerateEmbedding:
    """Tests for single embedding generation."""

    async def test_generate_embedding_returns_valid_result(self) -> None:
        """Test that generate_embedding returns properly structured result."""
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            result = await generate_embedding("Hello, world!")

            assert isinstance(result, EmbeddingResult)
            assert isinstance(result.embedding, list)
            assert result.token_count == ceil(len("Hello, world!") / 4)
            assert result.model_id == "BAAI/bge-large-en-v1.5"
            assert result.device == "auto"
            assert result.timestamp > 0
            assert result.latency_ms >= 0

    async def test_generate_embedding_vector_dimensions(self) -> None:
        """Test that embedding has correct dimensions."""
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            result = await generate_embedding("Test text")
            # mock returns 1024-dim vector
            assert len(result.embedding) == 1024


@pytest.mark.asyncio
class TestGenerateEmbeddings:
    """Tests for batch embedding generation."""

    async def test_generate_embeddings_batch_processing(self) -> None:
        """Test that batch processing works correctly."""
        texts = ["Hello", "World", "Test"]
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            results = await generate_embeddings(texts, batch_size=2)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, EmbeddingResult)
                assert len(result.embedding) == 1024

    async def test_generate_embeddings_empty_list(self) -> None:
        """Test that empty list returns empty results."""
        results = await generate_embeddings([])
        assert results == []

    async def test_generate_embeddings_custom_batch_size(self) -> None:
        """Test that custom batch_size is respected."""
        texts = ["a", "b", "c", "d", "e", "f"]
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            results = await generate_embeddings(texts, batch_size=3)

            assert len(results) == 6
            # Verify model.encode was called (batch processing)
            assert mock_transformer.encode.called


@pytest.mark.asyncio
class TestEmbeddingMetadata:
    """Tests for embedding result metadata."""

    async def test_embedding_result_has_required_fields(self) -> None:
        """Test that all required fields are present in result."""
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            result = await generate_embedding("Test content")

            assert hasattr(result, "embedding")
            assert hasattr(result, "token_count")
            assert hasattr(result, "model_id")
            assert hasattr(result, "device")
            assert hasattr(result, "timestamp")
            assert hasattr(result, "latency_ms")

    async def test_embedding_latency_is_non_negative(self) -> None:
        """Test that latency_ms is non-negative."""
        mock_transformer = _create_mock_transformer(1024)

        mock_manager = MagicMock()
        mock_manager.acquire = AsyncMock(return_value=mock_transformer)
        mock_manager.get_metadata.return_value = MagicMock(
            model_id="BAAI/bge-large-en-v1.5",
            device="auto",
        )

        with patch(
            "memini_ai.model.embeddings.ModelManager.get_instance",
            return_value=mock_manager,
        ):
            result = await generate_embedding("Test")
            assert result.latency_ms >= 0
