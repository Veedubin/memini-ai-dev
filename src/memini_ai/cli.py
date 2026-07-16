"""memini-ai CLI: embedded Postgres lifecycle + MCP server launcher.

Subcommands (design doc T8, ``docs/design/v1.0.0-embedded-pgembed-architecture.md``):

- ``memini-ai init``     — create the embedded data dir + state dir and start the
  embedded pgembed server, then print the connection URI. Idempotent.
- ``memini-ai status``   — read ``~/.memini-ai/pgembed/server.json`` and print a
  human-readable status. Never starts the server.
- ``memini-ai stop``     — request an explicit shutdown of the embedded server.
  ``--force`` stops even when other clients have active handles.
- ``memini-ai migrate``  — copy data from an external Postgres (``$MEMINI_DB_URL``
  or ``--from <url>``) into the embedded server via ``pg_dump`` / ``pg_restore``.

Default (no subcommand) preserves the pre-v1.0.0 MCP-server launcher behaviour
(``--stdio`` / ``--host`` / ``--port``), so existing ``memini-ai --stdio``
invocations and the ``[project.scripts]`` entry point keep working.

Pure stdlib ``argparse`` — no extra dependencies. Never interactive.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memini_ai.utils.logger import logger

# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("~/.local/share/memini-ai/pgembed/data").expanduser()
STATE_DIR = Path("~/.memini-ai/pgembed").expanduser()
STATE_FILE = STATE_DIR / "server.json"


def _resolve_data_dir() -> Path:
    """Resolve the embedded data dir from env, falling back to the default."""
    env_val = os.environ.get("MEMINI_PGEMBED_DATA_DIR")
    return Path(env_val).expanduser() if env_val else DEFAULT_DATA_DIR


# ── init ──────────────────────────────────────────────────────────────────────


async def _init() -> None:
    """Create data + state dirs, start the embedded server, print the URI.

    Idempotent: re-running on an already-initialised data dir just prints the
    URI and exits 0.
    """
    data_dir = _resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    from memini_ai.postgres.driver import EmbeddedPGDriver

    driver = EmbeddedPGDriver(data_dir)
    uri = await driver.get_uri()
    # Leave the server running: init is a "bring it up and report" command,
    # not a one-shot query. We intentionally do NOT call driver.shutdown().
    print("Embedded PostgreSQL started")
    print(f"  data dir: {data_dir}")
    print(f"  state:    {STATE_FILE}")
    print(f"  uri:      {uri}")


# ── status ────────────────────────────────────────────────────────────────────


def _format_uptime(started_at_iso: str | None) -> str:
    """Human-readable uptime from an ISO 8601 ``server_started_at`` value."""
    if not started_at_iso:
        return "unknown"
    try:
        started = datetime.fromisoformat(started_at_iso)
    except ValueError:
        return "unknown"
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - started
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _heartbeat_age(last_hb_iso: str) -> str:
    """Age of a client's last heartbeat, human-readable."""
    try:
        last = datetime.fromisoformat(last_hb_iso)
    except (ValueError, TypeError):
        return "unknown"
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - last
    return f"{int(delta.total_seconds())}s ago"


def _print_state(state: dict[str, Any], *, from_driver: bool) -> None:
    """Print a human-readable status block from a parsed server.json dict."""
    state_val = state.get("state", "unknown")
    pid = state.get("pid")
    started_at = state.get("server_started_at")
    uri = state.get("uri")
    data_dir = state.get("data_dir")
    clients: dict[str, Any] = state.get("clients", {}) or {}

    source = "driver" if from_driver else "server.json"
    print(f"Embedded PostgreSQL server ({source})")
    print(f"  state:         {state_val}")
    print(f"  pid:           {pid if pid is not None else 'unknown'}")
    print(f"  started at:    {started_at or 'unknown'}")
    print(f"  uptime:        {_format_uptime(started_at)}")
    print(f"  data dir:      {data_dir or 'unknown'}")
    print(f"  uri:           {uri or 'unknown'}")
    print(f"  clients:       {len(clients)}")
    for client_id, info in clients.items():
        short = client_id[:8] if isinstance(client_id, str) else client_id
        client_pid = info.get("pid") if isinstance(info, dict) else None
        last_hb = info.get("last_heartbeat") if isinstance(info, dict) else None
        role = info.get("role") if isinstance(info, dict) else None
        print(
            f"    - {short}  pid={client_pid}  role={role}  "
            f"last_heartbeat={_heartbeat_age(last_hb)}"
        )

    shutdown_by = state.get("shutdown_initiated_by")
    if shutdown_by:
        print(f"  drain initiated by: {shutdown_by}")
        print(f"  drain started at:    {state.get('shutdown_initiated_at')}")


async def _status() -> None:
    """Print embedded server status. Never starts the server.

    Prefers ``EmbeddedPGDriver.get_health_report()`` when a driver can be
    constructed against the existing data dir; otherwise reads ``server.json``
    directly. Missing file → friendly message + exit 0. Corrupt file → error
    + exit 1.
    """
    if not STATE_FILE.exists():
        print("No embedded server running")
        return

    # Try the driver health report first (richer, alive-client aware).
    try:
        from memini_ai.postgres.driver import EmbeddedPGDriver

        driver = EmbeddedPGDriver(_resolve_data_dir())
        # ``get_health_report`` is a pure read of server.json + heartbeat map;
        # it does NOT start the server. Guard the import though.
        report = driver.get_health_report()
        if report and report.get("state") is not None:
            # Reconstruct a state dict for the shared printer so the output is
            # uniform between the driver and raw-file paths.
            state_for_print: dict[str, Any] = {
                "state": report.get("state"),
                "pid": report.get("pid"),
                "uri": report.get("uri"),
                "data_dir": report.get("data_dir"),
                "server_started_at": report.get("server_started_at"),
                "shutdown_initiated_by": report.get("shutdown_initiated_by"),
                "shutdown_initiated_at": report.get("shutdown_initiated_at"),
                "clients": {},  # health report does not enumerate per-client PIDs
            }
            # Fall through to raw file read for the per-client breakdown.
            try:
                raw = json.loads(STATE_FILE.read_text())
                if isinstance(raw, dict):
                    state_for_print["clients"] = raw.get("clients", {}) or {}
            except (json.JSONDecodeError, OSError):
                pass
            _print_state(state_for_print, from_driver=True)
            return
    except Exception as e:  # noqa: BLE001 - status must never crash the CLI
        logger.debug("status_driver_unavailable", error=str(e))

    # Fallback: read server.json directly.
    try:
        raw: Any = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: corrupted server.json ({e})", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: cannot read {STATE_FILE}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(raw, dict):
        print("Error: corrupted server.json (not a JSON object)", file=sys.stderr)
        sys.exit(1)
    _print_state(raw, from_driver=False)


# ── stop ──────────────────────────────────────────────────────────────────────


async def _stop(*, force: bool) -> None:
    """Stop the embedded server.

    Safe to run when no server is running. ``--force`` stops even when other
    clients have active handles (uses ``cleanup_mode='stop'`` instead of
    ``None`` so pgembed itself tears the server down).
    """
    if not STATE_FILE.exists():
        print("No embedded server running")
        return

    from memini_ai.postgres.driver import EmbeddedPGDriver

    data_dir = _resolve_data_dir()
    driver = EmbeddedPGDriver(data_dir)

    # Attach to the existing server so we can shut it down. With --force we
    # construct via pgembed.get_server(cleanup_mode='stop') so pgembed tears the
    # server down on its own cleanup, bypassing the "last client" check.
    if force:
        import pgembed  # local import; only needed for the force path

        try:
            server = pgembed.get_server(str(data_dir), cleanup_mode="stop")
        except Exception as e:  # noqa: BLE001 - surface a clear CLI error
            print(f"Error: could not attach to embedded server: {e}", file=sys.stderr)
            sys.exit(1)
        # Ensure our state file reflects the stop.
        server.cleanup()
        # Mark state stopped if the driver's _stop_server didn't already.
        try:
            raw = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
            if isinstance(raw, dict) and raw.get("state") != "stopped":
                raw["state"] = "stopped"
                STATE_FILE.write_text(json.dumps(raw, indent=2))
        except (OSError, json.JSONDecodeError):
            pass
        print("Embedded PostgreSQL server stopped (forced)")
        return

    # Normal path: cooperative shutdown through the driver.
    try:
        await driver.get_uri()
    except Exception as e:  # noqa: BLE101 - server may already be stopped/dead
        print(f"No running embedded server to stop ({e})", file=sys.stderr)
        return

    driver.request_explicit_shutdown()
    await driver.shutdown()

    # Verify the state file now shows "stopped".
    try:
        raw = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        raw = {}
    final_state = raw.get("state") if isinstance(raw, dict) else None
    if final_state == "stopped":
        print("Embedded PostgreSQL server stopped")
    else:
        # shutdown() may have left "running" if other clients are still alive.
        print(
            f"Embedded PostgreSQL server shutdown requested (state={final_state})",
        )


# ── migrate ──────────────────────────────────────────────────────────────────

# Tables to verify after migration (kept in sync with the standalone script
# at ``scripts/migrate_external_to_embedded.py``). The actual verification
# reads the live table list from the source so this is a minimum set, not a
# hardcode — see ``_verify_migration``.
_VERIFICATION_TABLES: tuple[str, ...] = (
    "memories",
    "thought_chains",
    "thoughts",
    "audit_log",
    "memory_relationships",
    "trust_adjustments",
    "entities",
    "entity_relationships",
    "memory_sharing",
    "peers",
    "user_profiles",
    "memories_1024",
    "memories_image",
)

# Extensions that the embedded pgembed target does NOT ship — must be excluded
# from the dump so ``pg_restore`` does not fail with "extension ... not available".
_EXCLUDED_EXTENSIONS: tuple[str, ...] = ("timescaledb", "timescaledb_toolkit")

# Extensions that pgembed DOES ship and the dump will try to CREATE — we
# pre-install them so the ``CREATE EXTENSION`` statements in the dump become
# no-ops (``pg_restore --clean --if-exists`` will still try to recreate them).
_PREINSTALL_EXTENSIONS: tuple[str, ...] = ("vector", "vectorscale")


def _pgembed_bin_dir() -> Path:
    """Locate the bundled pgembed ``pginstall/bin`` directory.

    Falls back to PATH lookup if pgembed is not importable from this process.
    """
    try:
        import pgembed
    except ImportError:
        return Path(".")
    return Path(pgembed.__file__).resolve().parent / "pginstall" / "bin"


def _pg_option(uri: str, key: str, *, default: str = "") -> str:
    """Extract a query-string option from a postgres URI."""
    # postgresql://user:pass@host:port/db?host=/path&...
    if "?" not in uri:
        return default
    query = uri.split("?", 1)[1]
    for pair in query.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k == key:
                return v
    return default


def _pg_dbname(uri: str) -> str:
    """Extract the database name from a postgres URI (path component)."""
    # Strip scheme + authority.
    rest = uri
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    # Strip query.
    if "?" in rest:
        rest = rest.split("?", 1)[0]
    # Strip authority.
    if "/" in rest:
        rest = rest.split("/", 1)[1]
    return rest or "postgres"


def _vectors_equal(a: object, b: object) -> bool:
    """Compare two pgvector values (list[float] or str) for equality."""
    if a == b:
        return True
    a_list = _to_float_list(a)
    b_list = _to_float_list(b)
    if a_list is None or b_list is None:
        return False
    if len(a_list) != len(b_list):
        return False
    return all(abs(x - y) < 1e-9 for x, y in zip(a_list, b_list, strict=True))


def _to_float_list(v: object) -> list[float] | None:
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str):
        s = v.strip().lstrip("[").rstrip("]")
        if not s:
            return []
        try:
            return [float(x) for x in s.split(",")]
        except ValueError:
            return None
    return None


async def _preinstall_extensions(target_uri: str) -> None:
    """Pre-install ``vector`` + ``vectorscale`` on the target before restore."""
    import asyncpg  # local import; only needed for migrate

    conn = await asyncpg.connect(target_uri)
    try:
        for ext in _PREINSTALL_EXTENSIONS:
            try:
                await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                print(f"  pre-installed extension: {ext}")
            except Exception as e:  # noqa: BLE001 - warn, don't abort
                print(f"  WARNING: could not pre-install {ext}: {e}", file=sys.stderr)
    finally:
        await conn.close()


async def _count_rows(db_uri: str, table: str) -> int:
    """Return ``SELECT count(*) FROM <table>`` or -1 if the table is missing."""
    import asyncpg

    conn = await asyncpg.connect(db_uri)
    try:
        try:
            return int(await conn.fetchval(f"SELECT count(*) FROM {table}"))
        except asyncpg.UndefinedTableError:
            return -1
    finally:
        await conn.close()


async def _list_tables(db_uri: str) -> list[str]:
    """Return the list of base tables in the public schema."""
    import asyncpg

    conn = await asyncpg.connect(db_uri)
    try:
        rows = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE' "
            "ORDER BY table_name"
        )
        return [r["table_name"] for r in rows]
    finally:
        await conn.close()


async def _verify_migration(source_url: str, target_uri: str) -> bool:
    """Post-restore verification: row counts, spot-check, index check.

    Uses the actual table list from the source (not a hardcode) and the
    correct column name ``text`` (not ``content``).
    """
    import asyncpg

    tables = await _list_tables(source_url)
    if not tables:
        tables = list(_VERIFICATION_TABLES)

    src_counts: dict[str, int] = {}
    tgt_counts: dict[str, int] = {}

    src_conn = await asyncpg.connect(source_url)
    try:
        for t in tables:
            try:
                src_counts[t] = int(
                    await src_conn.fetchval(f"SELECT count(*) FROM {t}")
                )
            except asyncpg.UndefinedTableError:
                src_counts[t] = -1
    finally:
        await src_conn.close()

    conn = await asyncpg.connect(target_uri)
    try:
        for t in tables:
            try:
                tgt_counts[t] = int(await conn.fetchval(f"SELECT count(*) FROM {t}"))
            except asyncpg.UndefinedTableError:
                tgt_counts[t] = -1

        # Spot-check: pick a random memory and compare text + embedding exactly.
        spot_ok = False
        spot_detail = ""
        try:
            row = await conn.fetchrow(
                "SELECT id, text, embedding FROM memories ORDER BY random() LIMIT 1"
            )
            if row is not None:
                mem_id = row["id"]
                src2 = await asyncpg.connect(source_url)
                try:
                    src_row = await src2.fetchrow(
                        "SELECT text, embedding FROM memories WHERE id = $1",
                        mem_id,
                    )
                finally:
                    await src2.close()
                if src_row is not None:
                    text_match = row["text"] == src_row["text"]
                    emb_match = _vectors_equal(row["embedding"], src_row["embedding"])
                    spot_ok = text_match and emb_match
                    spot_detail = (
                        f"memory {mem_id}: text_match={text_match} "
                        f"embedding_match={emb_match}"
                    )
                else:
                    spot_detail = f"memory {mem_id} not found on source"
            else:
                spot_detail = "no memories on target to spot-check"
        except Exception as e:  # noqa: BLE001
            spot_detail = f"spot-check error: {e}"

        # Index check: confirm diskann indexes exist on the target.
        diskann_rows = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname='public' AND indexdef ILIKE '%USING diskann%' "
            "ORDER BY indexname"
        )
        diskann_names = [r["indexname"] for r in diskann_rows]
    finally:
        await conn.close()

    print("\n--- Verification ---")
    all_counts_ok = True
    for t in tables:
        s = src_counts.get(t, -1)
        g = tgt_counts.get(t, -1)
        ok = s == g
        if not ok:
            all_counts_ok = False
        status = "OK" if ok else "MISMATCH"
        if s == -1:
            status = "source-missing"
        elif g == -1:
            status = "target-missing"
            all_counts_ok = False
        print(f"  {t:24s} source={s:>6} target={g:>6}  [{status}]")

    print(f"  spot-check: {'PASS' if spot_ok else 'FAIL'} ({spot_detail})")
    print(f"  diskann indexes on target: {len(diskann_names)}")
    for n in diskann_names:
        print(f"    - {n}")

    passed = all_counts_ok and spot_ok and len(diskann_names) > 0
    print(f"\nVerification: {'PASS' if passed else 'FAIL'}")
    return passed


async def _migrate(*, source_url: str | None, dry_run: bool = False) -> None:
    """Copy data from an external Postgres to the embedded server.

    Uses ``pg_dump`` on the source and ``pg_restore`` on the target — standard
    Postgres tooling. Does NOT delete the source data. Auto-starts the
    embedded server if needed.

    Bug fixes (parity with ``scripts/migrate_external_to_embedded.py`` v1.0.1):
    - ``pg_dump`` resolves from the system PATH (must be >= source server
      version; pgembed's pg17 ``pg_dump`` aborts on a pg18 source).
    - ``pg_restore`` resolves from pgembed's bundled pg17 binaries (matches
      the pg17 embedded target).
    - Pre-installs ``vector`` + ``vectorscale`` on the target before restore.
    - Excludes ``timescaledb`` + ``timescaledb_toolkit`` from the dump.
    - ``pg_restore`` uses ``check=False`` and only fails on real ``ERROR:``
      lines (ignoring timescaledb/role warnings).
    - ``--dry-run`` skips the restore and just counts rows.
    - Post-restore verification (row counts, random memory spot-check, diskann
      index check) prints a clear PASS/FAIL summary.
    """
    url = source_url or os.environ.get("MEMINI_DB_URL")
    if not url:
        print(
            "Error: no source database. Set $MEMINI_DB_URL or pass --from <url>.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve client binaries.
    # pg_dump: prefer SYSTEM (must be >= source server version; source is
    #   typically pg18 while pgembed ships pg17, and pg17 pg_dump refuses
    #   to dump a pg18 server).
    # pg_restore: prefer PGEMBED (must match the pg17 target major version).
    pgembed_bin_dir = _pgembed_bin_dir()
    pg_restore_bin = pgembed_bin_dir / "pg_restore"
    if not (pg_restore_bin.exists() and os.access(pg_restore_bin, os.X_OK)):
        # Fall back to PATH if pgembed not importable / binary missing.
        pg_restore_bin = Path("pg_restore")

    # Start (or attach to) the embedded server to get the target URI.
    from memini_ai.postgres.driver import EmbeddedPGDriver

    data_dir = _resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    driver = EmbeddedPGDriver(data_dir)

    def _stop_driver() -> None:
        """Cooperative shutdown — request_explicit_shutdown is SYNC (do not await)."""
        with contextlib.suppress(Exception):
            driver.request_explicit_shutdown()

    try:
        target_uri = await driver.get_uri()
    except (OSError, RuntimeError) as e:
        print(f"Error: could not start embedded Postgres: {e}", file=sys.stderr)
        sys.exit(1)

    print("Migrating data")
    print(f"  source (external): {url}")
    print(f"  target (embedded): {target_uri}")
    print("  pg_dump:           system PATH (pg_dump)")
    print(f"  pg_restore:        {pg_restore_bin}")
    if dry_run:
        print("  mode:              DRY RUN (no restore)")

    # Pre-install extensions on the target before restore.
    print("  pre-installing extensions on target ...")
    await _preinstall_extensions(target_uri)

    # Count source + target rows (useful for both dry-run and real run).
    print("  counting source rows ...")
    source_tables = await _list_tables(url)
    src_counts: dict[str, int] = {}
    for t in source_tables:
        src_counts[t] = await _count_rows(url, t)
        print(f"    source {t}: {src_counts[t]}")

    print("  counting target rows (post pre-install, pre-restore) ...")
    for t in source_tables:
        c = await _count_rows(target_uri, t)
        print(f"    target {t}: {c}")

    # Build the dump command. pg_dump uses the system binary (>= source version).
    dump_cmd: list[str] = [
        "pg_dump",
        "--no-owner",
        "--no-privileges",
        "--format=custom",
        f"--dbname={url}",
    ]
    for ext in _EXCLUDED_EXTENSIONS:
        dump_cmd.append(f"--exclude-extension={ext}")

    print("  running pg_dump ...")
    try:
        dump_proc = subprocess.run(dump_cmd, capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: pg_dump not found on PATH", file=sys.stderr)
        _stop_driver()
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: pg_dump failed (exit {e.returncode}):\n"
            f"{e.stderr.decode(errors='replace') if e.stderr else ''}",
            file=sys.stderr,
        )
        _stop_driver()
        sys.exit(1)

    print(f"  pg_dump produced {len(dump_proc.stdout)} bytes")

    if dry_run:
        print("\nDRY RUN: skipping restore. Source row counts printed above.")
        print("Dry-run complete. No data was restored.")
        _stop_driver()
        return

    # Build the restore command using pgembed's bundled pg_restore (pg17).
    target_host = _pg_option(target_uri, "host")
    target_port = _pg_option(target_uri, "port", default="5432")
    target_db = _pg_dbname(target_uri)
    restore_cmd: list[str] = [
        str(pg_restore_bin),
        "--no-owner",
        "--no-privileges",
        "--clean",
        "--if-exists",
        f"--host={target_host}",
        f"--port={target_port}",
        f"--dbname={target_db}",
        "--username=postgres",
    ]

    print("  running pg_restore ...")
    # pg_restore returns nonzero for harmless errors (role mismatches, missing
    # extensions, etc). Use check=False and inspect stderr for real ERROR: lines.
    restore_proc = subprocess.run(
        restore_cmd,
        input=dump_proc.stdout,
        capture_output=True,
        check=False,
    )
    stderr_text = (
        restore_proc.stderr.decode(errors="replace") if restore_proc.stderr else ""
    )

    real_errors = [
        line
        for line in stderr_text.splitlines()
        if "ERROR:" in line
        and "timescaledb" not in line.lower()
        and not _is_role_error(line)
    ]
    if real_errors:
        print(
            "Error: pg_restore had real errors:\n" + "\n".join(real_errors),
            file=sys.stderr,
        )
        print(f"\nFull stderr:\n{stderr_text}", file=sys.stderr)
        _stop_driver()
        sys.exit(1)
    if restore_proc.returncode != 0:
        # Harmless warnings only — surface a short excerpt for transparency.
        print(f"  pg_restore completed with warnings (rc={restore_proc.returncode})")
        if stderr_text:
            print(f"  warnings: {stderr_text[:500]}")
    else:
        print("  pg_restore completed cleanly")

    print("Migration complete (source data left intact)")

    # Verification step (Bug 4): row counts, spot-check, index check.
    verified = await _verify_migration(url, target_uri)

    _stop_driver()
    if not verified:
        print(
            "\nMigration finished but verification FAILED — see above.",
            file=sys.stderr,
        )
        sys.exit(1)


def _is_role_error(line: str) -> bool:
    """True if a pg_restore stderr line is a harmless role/ownership error."""
    lower = line.lower()
    return ('role "' in lower and "does not exist" in lower) or (
        'role "' in lower and "already exists" in lower
    )


# ── MCP server launcher (default, no subcommand) ─────────────────────────────


def _run_server(args: argparse.Namespace) -> None:
    """Backwards-compatible MCP server launcher (pre-v1.0.0 ``main.py``).

    ``memini-ai --stdio`` and ``memini-ai --host ... --port ...`` keep working
    unchanged after the entry point moved to ``memini_ai.cli:main``.
    """
    from memini_ai.server import server

    if args.stdio:
        logger.info("starting_mcp_stdio")
        server.run(transport="stdio")
    else:
        logger.info("starting_mcp_http", host=args.host, port=args.port)
        try:
            server.run(transport="streamable-http", host=args.host, port=args.port)
        except KeyboardInterrupt:
            logger.info("server_interrupted")
            sys.exit(0)


# ── entry point ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memini-ai",
        description="memini-ai CLI: embedded Postgres lifecycle + MCP server.",
    )
    # Server-launch flags live on the TOP-LEVEL parser so ``memini-ai --stdio``
    # (no subcommand) keeps working. They are ignored when a subcommand is used.
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the MCP HTTP server to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the MCP HTTP server to (default: 8765).",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as a stdio MCP server instead of HTTP (no subcommand).",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "init", help="Initialize the embedded Postgres data dir and start the server."
    )
    subparsers.add_parser(
        "status", help="Show embedded server status (never starts the server)."
    )

    p_stop = subparsers.add_parser("stop", help="Stop the embedded Postgres server.")
    p_stop.add_argument(
        "--force",
        action="store_true",
        help="Force-stop even when other clients have active handles.",
    )

    p_migrate = subparsers.add_parser(
        "migrate",
        help="Migrate data from an external Postgres to the embedded server.",
    )
    p_migrate.add_argument(
        "--from",
        dest="source_url",
        help="Source MEMINI_DB_URL (default: $MEMINI_DB_URL).",
    )
    p_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Dump + count source rows + count target rows, but do NOT restore.",
    )

    return parser


def main() -> None:
    """CLI entry point (registered as ``memini-ai`` in pyproject.toml)."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "init":
        asyncio.run(_init())
    elif args.command == "status":
        asyncio.run(_status())
    elif args.command == "stop":
        asyncio.run(_stop(force=args.force))
    elif args.command == "migrate":
        asyncio.run(_migrate(source_url=args.source_url, dry_run=args.dry_run))
    elif args.command is None:
        # No subcommand: run the MCP server launcher (pre-v1.0.0 behaviour).
        _run_server(args)
    else:  # pragma: no cover - argparse rejects unknown subcommands
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
