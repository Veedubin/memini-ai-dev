"""Pytest fixtures for memini-ai tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from memini_ai.postgres.database import PostgresDatabase
from memini_ai.postgres.driver import EmbeddedPGDriver, ExternalPGDriver


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


@pytest_asyncio.fixture
async def pg_db(request, tmp_path):
    """PostgresDatabase fixture, parameterized on backend.

    Use indirect parametrization::

        @pytest.mark.parametrize("pg_db", ["external", "pgembed"], indirect=True)

    Or rely on the default (``"external"``) for backward-compatible local dev.
    """
    backend = getattr(request, "param", "external")
    if backend == "pgembed":
        data_dir = tmp_path / "pgembed_data"
        state_dir = tmp_path / "pgembed_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        driver = EmbeddedPGDriver(data_dir=data_dir, state_dir=state_dir)
    else:
        db_url = os.environ.get("MEMINI_DB_URL")
        if not db_url:
            pytest.skip("MEMINI_DB_URL not set — skipping external backend test")
        driver = ExternalPGDriver(db_url)
    db = PostgresDatabase(driver=driver)
    await db.initialize()
    yield db
    await db.close()
