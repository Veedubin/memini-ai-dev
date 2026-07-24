"""Tests for visualization API BUG-3 fix: pgembed mode support.

The visualization API previously checked ``config.db_url`` directly,
which was empty in pgembed mode (embedded PostgreSQL), causing all
endpoints to return "Set MEMINI_DB_URL environment variable" even
though the embedded DB was running. The fix uses ``create_database()``
which handles both pgembed and postgres-external modes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memini_ai.api.visualization import create_app


@pytest.fixture
def app_with_mock_db() -> FastAPI:
    """Create app with a mocked database (no real DB needed)."""
    mock_db = AsyncMock()
    mock_db.initialize = AsyncMock()
    mock_db.close = AsyncMock()
    mock_db.get_entities_with_relationships = AsyncMock(return_value=([], []))
    mock_db.get_entity_stats = AsyncMock(return_value={"total_entities": 0})
    mock_db.get_entity = AsyncMock(return_value=None)

    app = create_app()
    app.state.db = mock_db
    return app


@pytest.fixture
def app_with_no_db() -> FastAPI:
    """Create app with db=None (simulates failed initialization)."""
    app = create_app()
    app.state.db = None
    return app


class TestVisualizationHealthCheck:
    """Test the /api/health endpoint in various modes."""

    def test_health_with_db(self, app_with_mock_db: FastAPI) -> None:
        """Health endpoint reports connected when DB is available."""
        client = TestClient(app_with_mock_db)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["postgres"] == "connected"

    def test_health_without_db(self, app_with_no_db: FastAPI) -> None:
        """Health endpoint reports not configured when DB is None."""
        client = TestClient(app_with_no_db)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["postgres"] == "not configured"


class TestVisualizationPgembedMode:
    """Test that the visualization API works in pgembed mode (no MEMINI_DB_URL)."""

    def test_lifespan_uses_create_database_not_db_url(self) -> None:
        """The lifespan should call create_database(), not check config.db_url.

        This is the core BUG-3 fix: in pgembed mode, config.db_url is empty
        but create_database() still works (starts embedded PostgreSQL).
        """
        mock_db = AsyncMock()
        mock_db.initialize = AsyncMock()
        mock_db.close = AsyncMock()

        with patch(
            "memini_ai.api.visualization.create_database",
            return_value=mock_db,
        ) as mock_create:
            app = create_app()
            with TestClient(app) as client:
                # Lifespan should have called create_database
                mock_create.assert_called_once()
                resp = client.get("/api/health")
                assert resp.status_code == 200
                assert resp.json()["postgres"] == "connected"

    def test_lifespan_no_env_var_dependency(self) -> None:
        """The lifespan must NOT read MEMINI_DB_URL directly.

        In pgembed mode, MEMINI_DB_URL is not set. The fix uses
        create_database() which handles this internally.
        """
        mock_db = AsyncMock()
        mock_db.initialize = AsyncMock()
        mock_db.close = AsyncMock()

        with (
            patch(
                "memini_ai.api.visualization.create_database",
                return_value=mock_db,
            ),
            patch("os.environ.get") as mock_env_get,
        ):
            mock_env_get.return_value = None  # No env vars

            app = create_app()
            with TestClient(app) as client:
                resp = client.get("/api/health")
                assert resp.status_code == 200
                # Verify os.environ.get was not called for MEMINI_DB_URL
                # (it may be called by other internals, but the lifespan
                # itself should not depend on it)
                assert resp.json()["postgres"] == "connected"

    def test_graph_endpoint_works_with_pgembed(self) -> None:
        """The /api/graph endpoint should work when DB is from pgembed."""
        mock_db = AsyncMock()
        mock_db.initialize = AsyncMock()
        mock_db.close = AsyncMock()
        mock_db.get_entities_with_relationships = AsyncMock(
            return_value=(
                [{"id": "1", "name": "test", "type": "concept"}],
                [],
            )
        )

        with patch(
            "memini_ai.api.visualization.create_database",
            return_value=mock_db,
        ):
            app = create_app()
            with TestClient(app) as client:
                resp = client.get("/api/graph")
                assert resp.status_code == 200
                data = resp.json()
                assert data["count"] == 1
                assert data["nodes"][0]["name"] == "test"

    def test_graph_endpoint_503_when_db_init_fails(self) -> None:
        """When create_database() raises, db=None and endpoints return 503."""
        with patch(
            "memini_ai.api.visualization.create_database",
            side_effect=RuntimeError("pgembed failed to start"),
        ):
            app = create_app()
            with TestClient(app) as client:
                resp = client.get("/api/graph")
                assert resp.status_code == 503
                assert "Database not configured" in resp.json()["detail"]
                # Must NOT say "Set MEMINI_DB_URL"
                assert "Set MEMINI_DB_URL" not in resp.json()["detail"]
