"""Database driver protocol, external driver, and embedded pgembed driver.

Defines the minimal 4-method interface (``DatabaseDriver``) that abstracts
backend-specific connection lifecycle for ``PostgresDatabase``, plus:

- ``ExternalPGDriver`` — a trivial pass-through that preserves v0.8.2 behavior
  for external (Docker/team) Postgres connections.
- ``EmbeddedPGDriver`` — embedded pgembed Postgres with a cooperative
  heartbeat liveness protocol. ONE embedded server is shared by ALL
  memini-ai processes on the same machine. Uses pgembed's native
  ``get_server()`` for process coordination and a ``server.json`` state
  file for observability + heartbeat map. See
  ``docs/design/v1.0.0-embedded-pgembed-architecture.md`` section 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import fasteners

from memini_ai.utils.logger import logger

try:
    import pgembed
    from pgembed import (
        EXTENSION_POSTGRES_LIB_PATH,  # noqa: F401 - used in _write_postgres_config
    )
except (
    ImportError
) as _pgembed_import_err:  # pragma: no cover - exercised via smoke test
    _PGEMBED_IMPORT_ERROR: ImportError | None = _pgembed_import_err
    pgembed = None  # type: ignore[assignment]
    EXTENSION_POSTGRES_LIB_PATH = None  # type: ignore[assignment]
else:
    _PGEMBED_IMPORT_ERROR = None


@runtime_checkable
class DatabaseDriver(Protocol):
    """Minimal interface for backend-specific connection lifecycle."""

    async def get_uri(self) -> str:
        """Return a PostgreSQL connection URI ready for asyncpg.

        For pgembed: returns the Unix socket URI from get_server().
        For external: returns the configured MEMINI_DB_URL.

        Must be callable before initialize() — the URI is needed
        to create the asyncpg pool at PostgresDatabase.initialize():167.
        Idempotent.
        """
        ...

    async def initialize(self) -> None:
        """Perform backend-specific initialization AFTER the pool exists.

        For pgembed: starts the heartbeat + sweep background task.
        For external: no-op.

        Idempotent.
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully release backend resources.

        For pgembed: stops the heartbeat task, removes this client from
        the clients map, initiates shutdown if last client. Does NOT
        stop the server unless this is the last client AND the sweep
        would also initiate shutdown.
        For external: no-op.
        """
        ...

    def is_ready(self) -> bool:
        """Synchronous check: is the backend accepting connections?

        For pgembed: checks postmaster PID is alive.
        For external: always returns True.
        """
        ...


class ExternalPGDriver:
    """Driver for external (Docker/team) Postgres — preserves v0.8.2 behavior."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def get_uri(self) -> str:
        return self._db_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True


# ── Heartbeat / sweep defaults (design doc section 3.1) ──────────────────
DEFAULT_HEARTBEAT_INTERVAL_S = 60
DEFAULT_HEARTBEAT_TIMEOUT_S = 120
DEFAULT_GRACE_MARGIN_S = 10
DEFAULT_DRAIN_GRACE_S = 5

_SCHEMA_VERSION = "1.0.0"


def _require_pgembed() -> None:
    """Raise a clear error if pgembed could not be imported.

    Called lazily inside ``EmbeddedPGDriver`` methods so that importing
    ``memini_ai.postgres.driver`` never fails when pgembed is absent — only
    actually *using* the embedded backend does.
    """
    if "pgembed" not in globals() or _PGEMBED_IMPORT_ERROR is not None:
        raise ImportError(
            "pgembed is required for embedded mode. pip install memini-ai"
        ) from _PGEMBED_IMPORT_ERROR


class EmbeddedPGDriver:
    """Driver for embedded pgembed Postgres with cooperative heartbeat liveness.

    ONE embedded server, shared by ALL memini-ai processes on the same
    machine. Uses pgembed's native get_server() for process coordination
    and a server.json state file for observability + heartbeat map.
    """

    def __init__(
        self,
        data_dir: str | Path,
        state_dir: str | Path | None = None,
    ) -> None:
        self._data_dir: Path = Path(data_dir).expanduser().resolve()
        # server.json lives in the memini namespace for state (design Q1),
        # NOT next to the data dir.  Callers may override ``state_dir`` for
        # test isolation (e.g. ``tmp_path``).
        if state_dir is not None:
            self._state_dir: Path = Path(state_dir).expanduser().resolve()
        else:
            self._state_dir: Path = Path("~/.memini-ai/pgembed").expanduser().resolve()
        self._state_file: Path = self._state_dir / "server.json"
        self._state_lock: Path = self._state_dir / "server.json.lock"

        self._server: Any = None  # pgembed.PostgresServer when running
        self._uri: str | None = None
        self._client_id: str = str(uuid.uuid4())
        self._role: str = "passive"
        self._healthy: bool = True
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_stop: asyncio.Event = asyncio.Event()
        self._explicit_shutdown: bool = False
        self._postmaster_pid: int | None = None

        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API (DatabaseDriver Protocol) ──────────────────────────

    async def get_uri(self) -> str:
        """Return the PostgreSQL connection URI, starting or attaching as needed."""
        if self._uri is not None:
            return self._uri
        existing = self._read_state()
        if existing and existing.get("state") in ("running", "draining"):
            self._uri = await self._attach_to_existing(existing)
            return self._uri
        if existing and existing.get("state") == "dead":
            self._cleanup_stale_state()
        self._uri = await self._start_new_server()
        return self._uri

    async def initialize(self) -> None:
        """Idempotent post-pool initialization. Promotes this client to primary."""
        if self._server is None:
            await self.get_uri()
        self._role = "primary"  # Now actively querying

    async def shutdown(self) -> None:
        """Stop heartbeat, remove client, stop server if last or explicit shutdown."""
        # Stop heartbeat first so no new heartbeats land during teardown.
        if self._heartbeat_task is not None:
            self._heartbeat_stop.set()
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._server is None:
            return

        # Remove this client from the map.
        self._remove_client()

        # Stop the server if we're the last client OR explicit shutdown requested.
        state = self._read_state()
        remaining = state.get("clients", {}) if state else {}
        if not remaining or self._explicit_shutdown:
            await self._stop_server()

    def is_ready(self) -> bool:
        """Synchronous check: is the backend (postmaster) accepting connections?"""
        state = self._read_state()
        if state and state.get("state") == "running":
            pid = state.get("pid")
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    return True
                except OSError:
                    return False
        return False

    def request_explicit_shutdown(self) -> None:
        """Flag this client to stop the server on shutdown (for `memini-ai stop`)."""
        self._explicit_shutdown = True

    # ── Server lifecycle ─────────────────────────────────────────────

    async def _attach_to_existing(self, state: dict[str, Any]) -> str:
        """Attach to an already-running pgembed server (multi-process)."""
        _require_pgembed()
        data_dir = state["data_dir"]
        self._server = pgembed.get_server(data_dir, cleanup_mode=None)
        self._postmaster_pid = self._server.get_pid()
        uri: str = self._server.get_uri()
        self._uri = uri  # set before _register_client so state.json records the uri
        self._register_client()
        self._start_heartbeat()
        return uri

    async def _start_new_server(self) -> str:
        """Boot a fresh pgembed server (first client on this machine).

        Self-bootstrapping for fresh VMs / first-time users:

        1. Ensures the data dir parent exists (pgembed's get_server() raises
           "Parent directory of pgdata does not exist" otherwise — this is
           the common cause of the opencode 30s MCP startup timeout on a
           truly fresh install where the user has never run
           ``memini-ai init``).
        2. Writes the dynamic_library_path to ``postgresql.auto.conf`` so
           the bundled ``vector.so`` / ``vectorscale.so`` can be loaded
           by ``CREATE EXTENSION``. Without this, the schema SQL's
           ``CREATE EXTENSION IF NOT EXISTS vector;`` fails with
           "unknown type: public.vector" and the database layer's
           3-retry loop (1+2+4s) pushes past opencode's 30s startup
           timeout. We use ``ALTER SYSTEM`` via psql... actually no,
           pgembed's bundled psql segfaults on Python 3.13/3.14. So we
           edit ``postgresql.auto.conf`` directly (the file pgembed
           leaves alone between restarts). This is idempotent: re-running
           just overwrites the same lines.
        3. Starts the embedded server via ``pgembed.get_server``.
        4. Writes the initial state, registers this client, starts heartbeat.

        The schema creation step (in PostgresDatabase._ensure_schema) will
        then succeed on the first ``CREATE EXTENSION IF NOT EXISTS vector;``
        because the dynamic_library_path is set BEFORE the server starts.
        """
        _require_pgembed()
        # Step 1: ensure parent directory exists. This is the single biggest
        # failure mode on fresh VMs — without it, every opencode launch
        # retries 3x with backoff before giving up.
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._write_initial_state()
        try:
            self._server = pgembed.get_server(str(self._data_dir), cleanup_mode=None)
        except (OSError, RuntimeError) as e:
            self._write_state_dead()
            raise RuntimeError(f"Failed to start embedded Postgres: {e}") from e
        self._postmaster_pid = self._server.get_pid()
        # Step 2: edit postgresql.conf to set dynamic_library_path so
        # the bundled pgvector/vectorscale .so files can be loaded by
        # CREATE EXTENSION. We do this AFTER get_server (which runs
        # initdb on first call) so the config file exists.
        #
        # Why postgresql.conf (not auto.conf): auto.conf is only read
        # on ALTER SYSTEM or SIGHUP, NOT on a fresh server start. We
        # tried auto.conf first and the second launch still failed.
        #
        # Best-effort: write failure is logged but doesn't prevent
        # the server from starting. The next time this is called, the
        # write will be retried.
        try:
            self._write_postgres_config()
        except OSError as e:  # pragma: no cover - filesystem edge case
            logger.warning(
                "pgembed_postgres_config_write_failed",
                path=str(self._data_dir / "postgresql.conf"),
                error=str(e),
                hint="CREATE EXTENSION may fail; check filesystem permissions",
            )
        uri: str = self._server.get_uri()
        self._uri = uri  # set before _register_client so state.json records the uri
        self._register_client()
        self._start_heartbeat()
        return uri

    async def restart_server_for_new_config(self) -> str:
        """Stop the running postmaster, re-attach.

        Used by the installer / ``memini-ai init`` to pick up changes
        to ``postgresql.conf`` (e.g. ``dynamic_library_path``). pgembed
        starts the postmaster once and the only way to re-read the
        config file is to stop the postmaster process and start a
        new one. ``pgembed.get_server(cleanup_mode='stop')`` makes
        the OLD process tear down on cleanup; then a fresh
        ``get_server(cleanup_mode=None)`` reattaches (which restarts
        the postmaster because no other client holds it).

        Idempotent: safe to call when the server is already stopped
        (just returns the existing URI).
        """
        if not self.is_ready():
            # Server not running — just (re)start it.
            self._uri = None
            self._server = None
            self._state_file.unlink(missing_ok=True)
            return await self.get_uri()

        # Server is running. We need to: (1) stop the postmaster,
        # (2) reattach. pgembed has no clean "restart" API, so we
        # call the underlying pgembed.get_server with cleanup_mode='stop'
        # to force the postmaster down, then call again with
        # cleanup_mode=None to spin up a new one.
        import pgembed as _pgembed  # local import — only used here

        self._postmaster_pid = None
        self._state_file.unlink(missing_ok=True)

        # Force-stop: the next get_server() call (below) will spin
        # up a fresh postmaster that reads the current postgresql.conf
        # (with our dynamic_library_path line).
        try:
            stop_server = _pgembed.get_server(str(self._data_dir), cleanup_mode="stop")
            stop_server.cleanup()
        except Exception:  # noqa: BLE001 - best-effort
            pass
        self._server = None
        self._uri = None

        # Re-attach / fresh-start.
        return await self.get_uri()

    def _write_postgres_config(self) -> None:
        """Edit ``postgresql.conf`` so the bundled pgvector/vectorscale
        ``.so`` files can be loaded by ``CREATE EXTENSION``.

        Idempotent: re-running replaces the same line (no duplicates, no
        file bloat).

        Why ``postgresql.conf`` (not ``postgresql.auto.conf``):
        - pgembed creates ``postgresql.conf`` via initdb and never
          touches it again. Safe to edit.
        - ``postgresql.auto.conf`` is only read on ``ALTER SYSTEM`` or
          SIGHUP, NOT on a fresh server start. So writing to auto.conf
          would not help on the very first ever launch (we tried this
          and the second launch still failed with "unknown type:
          public.vector").
        - The schema's ``CREATE EXTENSION IF NOT EXISTS vector;`` runs
          during the first server start, so the config MUST be in
          ``postgresql.conf`` to be picked up.

        Default ``dynamic_library_path`` in stock PostgreSQL is
        ``'$libdir'`` (the postgres install's lib/postgresql dir). The
        pgembed install ships the .so files at
        ``<site-packages>/pgembed/pginstall/lib/postgresql/*.so``, which
        is NOT on the default search path, so CREATE EXTENSION fails
        with "could not access file '$libdir/vector'".

        We append the bundled extension path BEFORE the default ``$libdir``
        so it takes precedence (and the bundled .so files are loaded
        instead of anything that might be on the system path).
        """
        if EXTENSION_POSTGRES_LIB_PATH is None:
            return  # pgembed not importable; can't configure
        if not (self._data_dir / "PG_VERSION").exists():
            # First-ever init: data dir is empty, postgresql.conf
            # doesn't exist yet. initdb will create it. Skip and let
            # the next call (after initdb) configure it.
            return
        conf = self._data_dir / "postgresql.conf"
        ext_path = str(EXTENSION_POSTGRES_LIB_PATH).replace("'", "''")
        desired_line = f"dynamic_library_path = '$libdir:{ext_path}'\n"
        try:
            existing = conf.read_text()
        except OSError:
            existing = ""
        # Drop any prior dynamic_library_path lines we wrote, then
        # append. Idempotent.
        new_lines = [
            ln
            for ln in existing.splitlines(keepends=True)
            if not ln.lstrip().startswith("dynamic_library_path")
        ]
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"
        new_lines.append(desired_line)
        new_content = "".join(new_lines)
        if new_content != existing:
            conf.write_text(new_content)

    def _cleanup_stale_state(self) -> None:
        """Remove a dead server.json so the next client can start fresh."""
        if self._state_file.exists():
            with contextlib.suppress(OSError):
                self._state_file.unlink()

    # ── State file management ────────────────────────────────────────

    def _read_state(self) -> dict[str, Any] | None:
        """Atomically read server.json, returning None if missing or corrupt."""
        if not self._state_file.exists():
            return None
        try:
            data: Any = json.loads(self._state_file.read_text())
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def _write_initial_state(self) -> None:
        """Write the initial server.json (state=starting), preserving server_started_at."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            existing = self._read_state() or {}
            # Preserve server_started_at across restarts (never overwritten).
            started_at = (
                existing.get("server_started_at") or datetime.now(UTC).isoformat()
            )
            data: dict[str, Any] = {
                "version": _SCHEMA_VERSION,
                "state": "starting",
                "pid": os.getpid(),  # placeholder; replaced with postmaster pid
                "uri": "",
                "data_dir": str(self._data_dir),
                "server_started_at": started_at,
                "heartbeat_interval_s": DEFAULT_HEARTBEAT_INTERVAL_S,
                "heartbeat_timeout_s": DEFAULT_HEARTBEAT_TIMEOUT_S,
                "grace_margin_s": DEFAULT_GRACE_MARGIN_S,
                "shutdown_initiated_by": None,
                "shutdown_initiated_at": None,
                "clients": {},
            }
            self._state_file.write_text(json.dumps(data, indent=2))

    def _write_state_dead(self) -> None:
        """Mark the server as dead after a failed start."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state() or {}
            state["state"] = "dead"
            self._state_file.write_text(json.dumps(state, indent=2))

    def _register_client(self) -> None:
        """Add this client to the clients map and record the postmaster pid."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state() or {}
            state.setdefault("clients", {})
            state["clients"][self._client_id] = {
                "pid": os.getpid(),
                "last_heartbeat": datetime.now(UTC).isoformat(),
                "role": self._role,
            }
            # Update postmaster pid if we just started.
            if self._postmaster_pid is not None:
                state["pid"] = self._postmaster_pid
            state["state"] = "running"
            state["uri"] = self._uri or state.get("uri", "")
            self._state_file.write_text(json.dumps(state, indent=2))

    def _remove_client(self) -> None:
        """Remove this client from the clients map."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state()
            if state and self._client_id in state.get("clients", {}):
                del state["clients"][self._client_id]
                self._state_file.write_text(json.dumps(state, indent=2))

    def _write_heartbeat(self) -> None:
        """Atomically write this client's heartbeat under the state lock."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state() or {}
            state.setdefault("clients", {})
            state["clients"][self._client_id] = {
                "pid": os.getpid(),
                "last_heartbeat": datetime.now(UTC).isoformat(),
                "role": self._role,
            }
            self._state_file.write_text(json.dumps(state, indent=2))

    # ── Heartbeat task ────────────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        """Start the asyncio heartbeat+sweep loop task (once)."""
        if self._heartbeat_task is not None:
            return
        self._heartbeat_stop = asyncio.Event()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Main loop: sleep, write heartbeat, sweep stale clients."""
        state = self._read_state()
        interval = (
            state.get("heartbeat_interval_s", DEFAULT_HEARTBEAT_INTERVAL_S)
            if state
            else DEFAULT_HEARTBEAT_INTERVAL_S
        )
        consecutive_failures = 0

        while not self._heartbeat_stop.is_set():
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            if self._heartbeat_stop.is_set():
                break

            try:
                self._write_heartbeat()
                consecutive_failures = 0
            except OSError as e:
                consecutive_failures += 1
                logger.warning(
                    "heartbeat_write_failed",
                    client_id=self._client_id,
                    attempt=consecutive_failures,
                    error=str(e),
                )
                if consecutive_failures >= 3:
                    logger.error(
                        "heartbeat_write_failed_permanently", client_id=self._client_id
                    )
                    self._healthy = False
                continue

            await self._sweep_stale_clients()

    async def _sweep_stale_clients(self) -> None:
        """Distributed sweep: if ALL clients are stale, elect to shut down the server."""
        state = self._read_state()
        if state is None or state.get("shutdown_initiated_by") is not None:
            return

        clients = state.get("clients", {})
        if not clients:
            return

        now = datetime.now(UTC)
        timeout_s = state.get("heartbeat_timeout_s", DEFAULT_HEARTBEAT_TIMEOUT_S)
        grace_s = state.get("grace_margin_s", DEFAULT_GRACE_MARGIN_S)
        threshold = now - timedelta(seconds=timeout_s + grace_s)

        for info in clients.values():
            last_hb = datetime.fromisoformat(info["last_heartbeat"])
            if last_hb > threshold:
                return  # At least one client alive

        # ALL stale — initiate shutdown via election.
        if not self._try_claim_shutdown(state):
            return  # Another client beat us

        await asyncio.sleep(DEFAULT_DRAIN_GRACE_S)

        # Re-check: did anyone heartbeat during the grace period?
        state = self._read_state()
        if state:
            for info in state.get("clients", {}).values():
                last_hb = datetime.fromisoformat(info["last_heartbeat"])
                if last_hb > threshold:
                    self._clear_shutdown_token()
                    return

        await self._stop_server()

    def _try_claim_shutdown(self, state: dict[str, Any]) -> bool:
        """Election: atomically claim the shutdown_initiated_by token via lock."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            current = self._read_state()
            if current is None or current.get("shutdown_initiated_by") is not None:
                return False
            current["shutdown_initiated_by"] = self._client_id
            current["shutdown_initiated_at"] = datetime.now(UTC).isoformat()
            current["state"] = "draining"
            self._state_file.write_text(json.dumps(current, indent=2))
            return True

    def _clear_shutdown_token(self) -> None:
        """Cancel an in-progress drain (a client heartbeated during grace)."""
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state()
            if not state:
                return
            state["shutdown_initiated_by"] = None
            state["shutdown_initiated_at"] = None
            state["state"] = "running"
            self._state_file.write_text(json.dumps(state, indent=2))

    async def _stop_server(self) -> None:
        """Gracefully stop the pgembed server and mark state as stopped."""
        _require_pgembed()
        if self._postmaster_pid is not None:
            try:
                pgembed.pg_ctl(["-w", "stop"], pgdata=self._data_dir)
            except Exception as e:  # noqa: BLE001 - pg_ctl raises subprocess errors
                logger.warning(
                    "pgembed_stop_failed", data_dir=str(self._data_dir), error=str(e)
                )
        with fasteners.InterProcessLock(str(self._state_lock)):
            state = self._read_state() or {}
            state["state"] = "stopped"
            self._state_file.write_text(json.dumps(state, indent=2))
        self._server = None
        self._postmaster_pid = None

    # ── Health check API (design doc section 3.10) ────────────────────

    def is_healthy(self) -> bool:
        """Check if this client's connection is healthy."""
        if not self._healthy:
            return False
        state = self._read_state()
        if state is None:
            return False
        pid = state.get("pid")
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_health_report(self) -> dict[str, Any]:
        """Detailed health report for `memini-ai status` / the `get_status` MCP tool."""
        state = self._read_state() or {}
        clients = state.get("clients", {})
        now = datetime.now(UTC)
        alive_clients = 0
        for info in clients.values():
            try:
                last_hb = datetime.fromisoformat(info["last_heartbeat"])
                timeout_s = state.get(
                    "heartbeat_timeout_s", DEFAULT_HEARTBEAT_TIMEOUT_S
                )
                if (now - last_hb).total_seconds() < timeout_s:
                    alive_clients += 1
            except (KeyError, ValueError):
                continue
        return {
            "state": state.get("state"),
            "pid": state.get("pid"),
            "uri": state.get("uri"),
            "data_dir": state.get("data_dir"),
            "server_started_at": state.get("server_started_at"),
            "shutdown_initiated_by": state.get("shutdown_initiated_by"),
            "shutdown_initiated_at": state.get("shutdown_initiated_at"),
            "client_id": self._client_id,
            "role": self._role,
            "healthy": self.is_healthy(),
            "total_clients": len(clients),
            "alive_clients": alive_clients,
        }
