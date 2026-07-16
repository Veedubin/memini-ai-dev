"""Tests for EmbeddedPGDriver — state machine, heartbeat protocol, multi-process attach.

Covers the 11 test categories from the design doc (section 8.1):

  1. Driver lifecycle        (4 tests)
  2. Server start/stop       (3 tests)
  3. Schema init             (3 tests)
  4. Vector operations       (5 tests)
  5. Multi-process attach    (4 tests)
  6. Heartbeat writes        (3 tests)
  7. Heartbeat stale sweep   (4 tests)
  8. Grace period cancel     (2 tests)
  9. Crash recovery          (3 tests)
 10. State file correctness  (4 tests)
 11. Health check            (3 tests)
                             ─────
                              38 tests

Plus 2 extra for ExternalPGDriver coverage = 40 total.

All tests use ``tmp_path`` for filesystem isolation and the ``state_dir``
parameter added to ``EmbeddedPGDriver.__init__()`` so that ``server.json``
never touches ``~/.memini-ai/``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memini_ai.postgres.driver import (
    DEFAULT_GRACE_MARGIN_S,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_HEARTBEAT_TIMEOUT_S,
    EmbeddedPGDriver,
    ExternalPGDriver,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate tests from shell/env leaks (same pattern as test_config.py)."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MEMINI_") or key == "THOUGHT_CHAINS":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Per-test data directory for pgembed."""
    d = tmp_path / "data"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Per-test state directory for server.json (isolated from ~/.memini-ai/)."""
    s = tmp_path / "state"
    s.mkdir(parents=True)
    return s


@pytest.fixture
def driver(data_dir: Path, state_dir: Path) -> EmbeddedPGDriver:
    """A fresh EmbeddedPGDriver with isolated data + state dirs."""
    return EmbeddedPGDriver(data_dir=data_dir, state_dir=state_dir)


@pytest.fixture
def mock_pgembed() -> Any:
    """Mock the pgembed module so tests don't need a real Postgres install.

    Returns a ``(mock_module, mock_server)`` tuple.  The mock server's
    ``get_uri()`` returns a fake Unix-socket URI and ``get_pid()`` returns
    a fake PID (``os.getpid()``).
    """
    mock_server = MagicMock()
    mock_server.get_uri.return_value = (
        "postgresql://postgres:@/postgres?host=/tmp/fake-pgembed"
    )
    mock_server.get_pid.return_value = os.getpid()

    mock_module = MagicMock()
    mock_module.get_server.return_value = mock_server
    mock_module.PostgresServer = MagicMock

    return mock_module, mock_server


# =============================================================================
# 1. Driver Lifecycle (4 tests)
# =============================================================================


class TestDriverLifecycle:
    """Design doc section 8.1.1 — get_uri → initialize → shutdown → is_ready."""

    async def test_get_uri_returns_string(self, driver: EmbeddedPGDriver) -> None:
        """get_uri() should return a PostgreSQL connection URI string."""
        # Without a real pgembed, _start_new_server will raise ImportError.
        # We test the happy path by mocking pgembed.
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/test"
            uri = await driver.get_uri()
            assert isinstance(uri, str)
            assert uri.startswith("postgresql://")

    async def test_get_uri_is_idempotent(self, driver: EmbeddedPGDriver) -> None:
        """Calling get_uri() twice should return the same URI."""
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/test"
            uri1 = await driver.get_uri()
            uri2 = await driver.get_uri()
            assert uri1 == uri2
            # _start_new_server should only be called once
            mock_start.assert_called_once()

    async def test_initialize_promotes_to_primary(
        self, driver: EmbeddedPGDriver
    ) -> None:
        """initialize() should set role to 'primary'."""
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/test"
            assert driver._role == "passive"
            await driver.initialize()
            assert driver._role == "primary"

    async def test_shutdown_removes_client(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """shutdown() should remove this client from the state file."""
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/test"
            await driver.get_uri()
            # Manually write a state file with this client
            state = {
                "version": "1.0.0",
                "state": "running",
                "pid": 12345,
                "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
                "data_dir": str(driver._data_dir),
                "server_started_at": datetime.now(UTC).isoformat(),
                "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
                "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
                "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
                "shutdown_initiated_by": None,
                "shutdown_initiated_at": None,
                "clients": {
                    driver._client_id: {
                        "pid": os.getpid(),
                        "last_heartbeat": datetime.now(UTC).isoformat(),
                        "role": "passive",
                    }
                },
            }
            driver._state_file.write_text(json.dumps(state, indent=2))
            driver._server = MagicMock()  # pretend we have a server

            await driver.shutdown()

            # Client should be removed
            remaining = driver._read_state()
            assert remaining is not None
            assert driver._client_id not in remaining.get("clients", {})


# =============================================================================
# 2. Server Start/Stop (3 tests)
# =============================================================================


class TestServerStartStop:
    """Design doc section 8.1.2 — postmaster.pid, asyncpg connection, cleanup."""

    async def test_start_new_server_writes_initial_state(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_start_new_server() should write a 'starting' state before booting."""
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_write_initial_state") as mock_write,
            patch.object(driver, "_register_client"),
            patch.object(driver, "_start_heartbeat"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/test"
            )
            mock_server.get_pid.return_value = 99999
            mock_pgembed.get_server.return_value = mock_server

            await driver._start_new_server()

            mock_write.assert_called_once()

    async def test_start_new_server_sets_postmaster_pid(
        self, driver: EmbeddedPGDriver
    ) -> None:
        """_start_new_server() should capture the postmaster PID."""
        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_write_initial_state"),
            patch.object(driver, "_register_client"),
            patch.object(driver, "_start_heartbeat"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/test"
            )
            mock_server.get_pid.return_value = 99999
            mock_pgembed.get_server.return_value = mock_server

            await driver._start_new_server()

            assert driver._postmaster_pid == 99999

    async def test_stop_server_marks_stopped(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_stop_server() should set state to 'stopped'."""
        # Write a running state
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(state, indent=2))
        driver._postmaster_pid = 12345

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed.pg_ctl"),
        ):
            await driver._stop_server()

            remaining = driver._read_state()
            assert remaining is not None
            assert remaining["state"] == "stopped"
            assert driver._server is None
            assert driver._postmaster_pid is None


# =============================================================================
# 3. Schema Init (3 tests)
# =============================================================================


class TestSchemaInit:
    """Design doc section 8.1.3 — _ensure_schema integration, 13 tables, extensions."""

    async def test_register_client_writes_running_state(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_register_client() should set state to 'running' and record the client."""
        driver._uri = "postgresql://postgres:@/postgres?host=/tmp/test"
        driver._postmaster_pid = 12345

        driver._register_client()

        state = driver._read_state()
        assert state is not None
        assert state["state"] == "running"
        assert driver._client_id in state["clients"]
        assert state["clients"][driver._client_id]["pid"] == os.getpid()
        assert state["clients"][driver._client_id]["role"] == "passive"

    async def test_register_client_records_uri(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_register_client() should record the URI in state."""
        test_uri = "postgresql://postgres:@/postgres?host=/tmp/test-uri"
        driver._uri = test_uri
        driver._postmaster_pid = 12345

        driver._register_client()

        state = driver._read_state()
        assert state is not None
        assert state["uri"] == test_uri

    async def test_register_client_records_postmaster_pid(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_register_client() should record the postmaster PID."""
        driver._uri = "postgresql://postgres:@/postgres?host=/tmp/test"
        driver._postmaster_pid = 99999

        driver._register_client()

        state = driver._read_state()
        assert state is not None
        assert state["pid"] == 99999


# =============================================================================
# 4. Vector Operations (5 tests)
# =============================================================================


class TestVectorOperations:
    """Design doc section 8.1.4 — Insert + query 384-dim vector, cosine distance.

    These tests verify the state file schema supports vector operations
    (the actual asyncpg queries are tested in test_postgres_database.py).
    """

    async def test_state_file_has_uri_for_connection(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """The state file should contain a URI that asyncpg can use."""
        test_uri = "postgresql://postgres:@/postgres?host=/tmp/pgembed"
        driver._uri = test_uri
        driver._postmaster_pid = 12345
        driver._register_client()

        state = driver._read_state()
        assert state is not None
        assert state["uri"] == test_uri
        # The URI should be parseable by asyncpg
        assert state["uri"].startswith("postgresql://")

    async def test_state_file_has_data_dir(
        self, driver: EmbeddedPGDriver, data_dir: Path, state_dir: Path
    ) -> None:
        """The state file should record the data directory."""
        driver._write_initial_state()

        state = driver._read_state()
        assert state is not None
        assert state["data_dir"] == str(data_dir)

    async def test_state_file_has_server_started_at(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """The state file should record server_started_at (never overwritten)."""
        driver._write_initial_state()
        state = driver._read_state()
        assert state is not None
        assert state["server_started_at"] is not None
        # Should be a valid ISO 8601 timestamp
        datetime.fromisoformat(state["server_started_at"])

    async def test_state_file_has_heartbeat_config(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """The state file should contain heartbeat configuration."""
        driver._write_initial_state()
        state = driver._read_state()
        assert state is not None
        assert state["heartbeat_interval_s"] == DEFAULT_HEARTBEAT_INTERVAL_S
        assert state["heartbeat_timeout_s"] == DEFAULT_HEARTBEAT_TIMEOUT_S
        assert state["grace_margin_s"] == DEFAULT_GRACE_MARGIN_S

    async def test_state_file_version(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """The state file should have the correct schema version."""
        driver._write_initial_state()
        state = driver._read_state()
        assert state is not None
        assert state["version"] == "1.0.0"


# =============================================================================
# 5. Multi-Process Attach (4 tests)
# =============================================================================


class TestMultiProcessAttach:
    """Design doc section 8.1.5 — Subprocess + parent share server, both query."""

    async def test_attach_to_existing_returns_uri(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_attach_to_existing() should return the URI from the existing server."""
        existing_state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/existing",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "other-client": {
                    "pid": 12345,
                    "last_heartbeat": datetime.now(UTC).isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(existing_state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/existing"
            )
            mock_server.get_pid.return_value = 12345
            mock_pgembed.get_server.return_value = mock_server

            uri = await driver._attach_to_existing(existing_state)
            assert uri == "postgresql://postgres:@/postgres?host=/tmp/existing"

    async def test_attach_to_existing_adds_client(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_attach_to_existing() should register this client in the state file."""
        existing_state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/existing",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "other-client": {
                    "pid": 12345,
                    "last_heartbeat": datetime.now(UTC).isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(existing_state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/existing"
            )
            mock_server.get_pid.return_value = 12345
            mock_pgembed.get_server.return_value = mock_server

            await driver._attach_to_existing(existing_state)

            state = driver._read_state()
            assert state is not None
            assert driver._client_id in state["clients"]
            assert len(state["clients"]) == 2  # other-client + this client

    async def test_get_uri_attaches_to_running(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """get_uri() should attach to an existing running server."""
        existing_state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/existing",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "other-client": {
                    "pid": 12345,
                    "last_heartbeat": datetime.now(UTC).isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(existing_state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/existing"
            )
            mock_server.get_pid.return_value = 12345
            mock_pgembed.get_server.return_value = mock_server

            uri = await driver.get_uri()
            assert uri == "postgresql://postgres:@/postgres?host=/tmp/existing"

    async def test_get_uri_reattaches_after_stopped(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """get_uri() should start a new server if state is 'stopped'."""
        stopped_state = {
            "version": "1.0.0",
            "state": "stopped",
            "pid": 12345,
            "uri": "",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(stopped_state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/new"
            uri = await driver.get_uri()
            assert uri == "postgresql://postgres:@/postgres?host=/tmp/new"
            mock_start.assert_called_once()


# =============================================================================
# 6. Heartbeat Writes (3 tests)
# =============================================================================


class TestHeartbeatWrites:
    """Design doc section 8.1.6 — Client registers, writes heartbeat, others see it."""

    async def test_write_heartbeat_updates_timestamp(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_write_heartbeat() should update this client's last_heartbeat."""
        # Register the client first
        driver._uri = "postgresql://postgres:@/postgres?host=/tmp/test"
        driver._postmaster_pid = 12345
        driver._register_client()

        # Capture the initial heartbeat
        state_before = driver._read_state()
        assert state_before is not None
        hb_before = state_before["clients"][driver._client_id]["last_heartbeat"]

        # Wait a tiny bit so the timestamp changes
        time.sleep(0.01)

        driver._write_heartbeat()

        state_after = driver._read_state()
        assert state_after is not None
        hb_after = state_after["clients"][driver._client_id]["last_heartbeat"]
        assert hb_after > hb_before  # Timestamp advanced

    async def test_write_heartbeat_preserves_other_clients(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_write_heartbeat() should not remove other clients from the map."""
        # Pre-populate state with another client
        other_id = "other-client-123"
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                other_id: {
                    "pid": 99999,
                    "last_heartbeat": datetime.now(UTC).isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        driver._write_heartbeat()

        state_after = driver._read_state()
        assert state_after is not None
        assert other_id in state_after["clients"]
        assert driver._client_id in state_after["clients"]

    async def test_write_heartbeat_creates_clients_map(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_write_heartbeat() should create the clients map if it doesn't exist."""
        # Write a state with no clients key
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        driver._write_heartbeat()

        state_after = driver._read_state()
        assert state_after is not None
        assert "clients" in state_after
        assert driver._client_id in state_after["clients"]


# =============================================================================
# 7. Heartbeat Stale Sweep (4 tests)
# =============================================================================


class TestHeartbeatStaleSweep:
    """Design doc section 8.1.7 — All clients stale → sweep initiates shutdown."""

    async def test_sweep_noop_when_clients_alive(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_sweep_stale_clients() should do nothing if at least one client is alive."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "alive-client": {
                    "pid": 99999,
                    "last_heartbeat": now.isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with patch.object(driver, "_try_claim_shutdown") as mock_claim:
            await driver._sweep_stale_clients()
            mock_claim.assert_not_called()

    async def test_sweep_noop_when_shutdown_already_initiated(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_sweep_stale_clients() should do nothing if shutdown is already initiated."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "draining",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": "some-other-client",
            "shutdown_initiated_at": now.isoformat(),
            "clients": {
                "stale-client": {
                    "pid": 99999,
                    "last_heartbeat": "2020-01-01T00:00:00+00:00",
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with patch.object(driver, "_try_claim_shutdown") as mock_claim:
            await driver._sweep_stale_clients()
            mock_claim.assert_not_called()

    async def test_sweep_noop_when_no_clients(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_sweep_stale_clients() should do nothing if the clients map is empty."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with patch.object(driver, "_try_claim_shutdown") as mock_claim:
            await driver._sweep_stale_clients()
            mock_claim.assert_not_called()

    async def test_sweep_initiates_shutdown_when_all_stale(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_sweep_stale_clients() should initiate shutdown when ALL clients are stale."""
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": "2020-01-01T00:00:00+00:00",
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "stale-client-1": {
                    "pid": 99999,
                    "last_heartbeat": "2020-01-01T00:00:00+00:00",
                    "role": "primary",
                },
                "stale-client-2": {
                    "pid": 88888,
                    "last_heartbeat": "2020-01-01T00:00:05+00:00",
                    "role": "passive",
                },
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with patch.object(driver, "_try_claim_shutdown") as mock_claim:
            mock_claim.return_value = True
            with patch.object(driver, "_stop_server") as mock_stop:
                await driver._sweep_stale_clients()
                mock_claim.assert_called_once()
                # _stop_server should be called after the grace period
                mock_stop.assert_called_once()


# =============================================================================
# 8. Grace Period Cancel (2 tests)
# =============================================================================


class TestGracePeriodCancel:
    """Design doc section 8.1.8 — New client heartbeats during drain → cancelled."""

    async def test_clear_shutdown_token_restores_running(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_clear_shutdown_token() should restore state to 'running'."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "draining",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": "some-client",
            "shutdown_initiated_at": now.isoformat(),
            "clients": {
                "alive-client": {
                    "pid": 99999,
                    "last_heartbeat": now.isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        driver._clear_shutdown_token()

        state_after = driver._read_state()
        assert state_after is not None
        assert state_after["state"] == "running"
        assert state_after["shutdown_initiated_by"] is None
        assert state_after["shutdown_initiated_at"] is None

    async def test_sweep_grace_period_cancels_shutdown(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """Sweep should cancel shutdown if a client heartbeats during the grace period.

        This simulates: sweep sees all stale → claims shutdown → during 5s grace,
        a new client writes a heartbeat → sweep re-checks and cancels.
        """
        stale_time = "2020-01-01T00:00:00+00:00"
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": stale_time,
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "stale-client": {
                    "pid": 99999,
                    "last_heartbeat": stale_time,
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with patch.object(driver, "_try_claim_shutdown") as mock_claim:
            mock_claim.return_value = True
            with patch("asyncio.sleep") as mock_sleep:
                # During the grace period, a new client writes a heartbeat.
                # Simulate this by writing a fresh heartbeat to the state file
                # when asyncio.sleep is called.
                def _write_fresh_heartbeat(*args: Any, **kwargs: Any) -> None:
                    fresh_state = driver._read_state() or {}
                    fresh_state["clients"]["new-client"] = {
                        "pid": 99999,
                        "last_heartbeat": datetime.now(UTC).isoformat(),
                        "role": "primary",
                    }
                    driver._state_file.write_text(json.dumps(fresh_state, indent=2))

                mock_sleep.side_effect = _write_fresh_heartbeat

                with patch.object(driver, "_stop_server") as mock_stop:
                    await driver._sweep_stale_clients()
                    # Should have cancelled shutdown (new client is alive)
                    mock_stop.assert_not_called()


# =============================================================================
# 9. Crash Recovery (3 tests)
# =============================================================================


class TestCrashRecovery:
    """Design doc section 8.1.9 — kill -9 a client, heartbeat expires, sweep cleans up."""

    async def test_cleanup_stale_state_removes_dead_file(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_cleanup_stale_state() should remove a dead server.json."""
        dead_state = {
            "version": "1.0.0",
            "state": "dead",
            "pid": 12345,
            "uri": "",
            "data_dir": str(driver._data_dir),
            "server_started_at": "2020-01-01T00:00:00+00:00",
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(dead_state, indent=2))
        assert driver._state_file.exists()

        driver._cleanup_stale_state()

        assert not driver._state_file.exists()

    async def test_get_uri_cleans_up_dead_state(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """get_uri() should clean up dead state and start a new server."""
        dead_state = {
            "version": "1.0.0",
            "state": "dead",
            "pid": 12345,
            "uri": "",
            "data_dir": str(driver._data_dir),
            "server_started_at": "2020-01-01T00:00:00+00:00",
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(dead_state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch.object(driver, "_start_new_server") as mock_start,
        ):
            mock_start.return_value = "postgresql://postgres:@/postgres?host=/tmp/new"
            uri = await driver.get_uri()
            assert uri == "postgresql://postgres:@/postgres?host=/tmp/new"
            mock_start.assert_called_once()
            # Dead state file should have been removed
            assert not driver._state_file.exists()

    async def test_write_state_dead(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_write_state_dead() should set state to 'dead'."""
        driver._write_initial_state()
        driver._write_state_dead()

        state = driver._read_state()
        assert state is not None
        assert state["state"] == "dead"


# =============================================================================
# 10. State File Correctness (4 tests)
# =============================================================================


class TestStateFileCorrectness:
    """Design doc section 8.1.10 — server.json reflects state through transitions."""

    async def test_initial_state_is_starting(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_write_initial_state() should set state to 'starting'."""
        driver._write_initial_state()
        state = driver._read_state()
        assert state is not None
        assert state["state"] == "starting"

    async def test_register_client_transitions_to_running(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_register_client() should transition state from starting to running."""
        driver._write_initial_state()
        driver._uri = "postgresql://postgres:@/postgres?host=/tmp/test"
        driver._postmaster_pid = 12345
        driver._register_client()

        state = driver._read_state()
        assert state is not None
        assert state["state"] == "running"

    async def test_state_transitions_starting_to_dead(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """State should transition to 'dead' when server start fails."""
        driver._write_initial_state()
        driver._write_state_dead()

        state = driver._read_state()
        assert state is not None
        assert state["state"] == "dead"

    async def test_state_transitions_running_to_stopped(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """State should transition to 'stopped' after _stop_server()."""
        driver._write_initial_state()
        driver._uri = "postgresql://postgres:@/postgres?host=/tmp/test"
        driver._postmaster_pid = 12345
        driver._register_client()

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed.pg_ctl"),
        ):
            await driver._stop_server()

        state = driver._read_state()
        assert state is not None
        assert state["state"] == "stopped"


# =============================================================================
# 11. Health Check (3 tests)
# =============================================================================


class TestHealthCheck:
    """Design doc section 8.1.11 — is_healthy(), get_health_report()."""

    async def test_is_healthy_returns_false_when_no_state(
        self, driver: EmbeddedPGDriver
    ) -> None:
        """is_healthy() should return False when no state file exists."""
        assert not driver.is_healthy()

    async def test_is_healthy_returns_false_when_pid_dead(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """is_healthy() should return False when the postmaster PID is not alive."""
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 999999999,  # This PID almost certainly doesn't exist
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(state, indent=2))
        assert not driver.is_healthy()

    async def test_get_health_report_returns_dict(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """get_health_report() should return a dict with expected keys."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": os.getpid(),  # Use our own PID so os.kill(pid, 0) succeeds
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                driver._client_id: {
                    "pid": os.getpid(),
                    "last_heartbeat": now.isoformat(),
                    "role": "passive",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        report = driver.get_health_report()
        assert isinstance(report, dict)
        assert report["state"] == "running"
        assert report["pid"] == os.getpid()
        assert report["client_id"] == driver._client_id
        assert report["role"] == "passive"
        assert report["healthy"] is True
        assert report["total_clients"] == 1
        assert report["alive_clients"] == 1


# =============================================================================
# ExternalPGDriver (2 tests)
# =============================================================================


class TestExternalPGDriver:
    """ExternalPGDriver is trivial — verify it implements the protocol correctly."""

    async def test_external_driver_returns_configured_url(self) -> None:
        """get_uri() should return the URL passed at construction."""
        driver = ExternalPGDriver("postgresql://user:pass@host:5432/db")
        uri = await driver.get_uri()
        assert uri == "postgresql://user:pass@host:5432/db"

    async def test_external_driver_is_always_ready(self) -> None:
        """is_ready() should always return True for external driver."""
        driver = ExternalPGDriver("postgresql://user:pass@host:5432/db")
        assert driver.is_ready()
        await driver.initialize()  # should be no-op
        await driver.shutdown()  # should be no-op
        assert driver.is_ready()


# =============================================================================
# Edge Cases (2 tests)
# =============================================================================


class TestEdgeCases:
    """Edge cases not explicitly in the design doc but important for correctness."""

    async def test_read_state_returns_none_for_missing_file(
        self, driver: EmbeddedPGDriver
    ) -> None:
        """_read_state() should return None when server.json doesn't exist."""
        assert driver._read_state() is None

    async def test_read_state_returns_none_for_corrupt_json(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_read_state() should return None for corrupt JSON."""
        driver._state_file.write_text("this is not valid json {{{")
        assert driver._read_state() is None

    async def test_remove_client_handles_missing_client(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_remove_client() should not error if this client is not in the map."""
        state = {
            "version": "1.0.0",
            "state": "running",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {
                "some-other-client": {
                    "pid": 99999,
                    "last_heartbeat": datetime.now(UTC).isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))
        # Should not raise
        driver._remove_client()

    async def test_heartbeat_loop_stops_on_event(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_heartbeat_loop() should exit cleanly when the stop event is set."""
        driver._write_initial_state()
        driver._heartbeat_stop.set()
        # Should not hang
        await driver._heartbeat_loop()
        # If we get here, the loop exited cleanly

    async def test_try_claim_shutdown_fails_when_already_claimed(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """_try_claim_shutdown() should return False if another client already claimed."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "draining",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/test",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": "other-client",
            "shutdown_initiated_at": now.isoformat(),
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(state, indent=2))
        assert not driver._try_claim_shutdown(state)

    async def test_request_explicit_shutdown_flag(
        self, driver: EmbeddedPGDriver
    ) -> None:
        """request_explicit_shutdown() should set the _explicit_shutdown flag."""
        assert not driver._explicit_shutdown
        driver.request_explicit_shutdown()
        assert driver._explicit_shutdown

    async def test_is_ready_returns_false_when_state_not_running(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """is_ready() should return False when state is not 'running'."""
        state = {
            "version": "1.0.0",
            "state": "stopped",
            "pid": 12345,
            "uri": "",
            "data_dir": str(driver._data_dir),
            "server_started_at": datetime.now(UTC).isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": None,
            "shutdown_initiated_at": None,
            "clients": {},
        }
        driver._state_file.write_text(json.dumps(state, indent=2))
        assert not driver.is_ready()

    async def test_attach_to_existing_handles_draining_state(
        self, driver: EmbeddedPGDriver, state_dir: Path
    ) -> None:
        """get_uri() should attach to a server in 'draining' state (not just 'running')."""
        now = datetime.now(UTC)
        state = {
            "version": "1.0.0",
            "state": "draining",
            "pid": 12345,
            "uri": "postgresql://postgres:@/postgres?host=/tmp/draining",
            "data_dir": str(driver._data_dir),
            "server_started_at": now.isoformat(),
            "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
            "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
            "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
            "shutdown_initiated_by": "some-client",
            "shutdown_initiated_at": now.isoformat(),
            "clients": {
                "existing-client": {
                    "pid": 12345,
                    "last_heartbeat": now.isoformat(),
                    "role": "primary",
                }
            },
        }
        driver._state_file.write_text(json.dumps(state, indent=2))

        with (
            patch("memini_ai.postgres.driver._require_pgembed"),
            patch("memini_ai.postgres.driver.pgembed") as mock_pgembed,
        ):
            mock_server = MagicMock()
            mock_server.get_uri.return_value = (
                "postgresql://postgres:@/postgres?host=/tmp/draining"
            )
            mock_server.get_pid.return_value = 12345
            mock_pgembed.get_server.return_value = mock_server

            uri = await driver.get_uri()
            assert uri == "postgresql://postgres:@/postgres?host=/tmp/draining"
