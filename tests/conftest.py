"""Pytest fixtures for memini-ai tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client."""
    client = MagicMock()
    client.search.return_value = []
    client.upsert.return_value = {"status": "completed"}
    client.delete.return_value = {"status": "completed"}
    client.get_collection.return_value = {"status": "green"}
    return client


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    """Create a mock sentence transformer model."""
    model = MagicMock()
    # Return proper 1024-dim vectors, one per input
    model.encode.return_value = [[0.1] * 1024]
    model.get_sentence_embedding_dimension.return_value = 1024
    return model


@pytest.fixture
def sample_memory_entry() -> dict[str, Any]:
    """Create a sample memory entry dict."""
    return {
        "id": "test-id-123",
        "text": "This is a test memory entry.",
        "vector": [0.1] * 1024,
        "sourceType": "session",
        "sourcePath": "/test/path",
        "contentHash": "abc123hash",
        "sessionId": "session-456",
        "projectId": "project-789",
    }