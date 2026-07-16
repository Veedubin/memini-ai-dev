#!/usr/bin/env python3
"""Migrate memini-ai data from an external Postgres to the embedded pgembed server.

Usage:
    memini-ai migrate --from=postgresql://user:pass@host:port/db
    memini-ai migrate  # uses $MEMINI_DB_URL as source
    memini-ai migrate --dry-run  # dump + pre-checks, no restore

This uses pg_dump + pg_restore under the hood. ``pg_dump`` is resolved from
the system PATH (it must be >= the source server version, which is typically
PostgreSQL 18 — pgembed's bundled pg17 ``pg_dump`` refuses to dump a pg18
server). ``pg_restore`` is resolved from pgembed's bundled binaries
(PostgreSQL 17) so it matches the pg17 embedded target. The embedded pgembed
server is started on demand and stopped at the end (cooperative heartbeat
means other memini-ai processes can keep using it if they are alive).

Bug fixes in v1.0.1 (see CHANGELOG):
- Use pgembed's bundled pg_dump/pg_restore (pg17), not the system pg18 binaries.
- Extract ?host= Unix socket param from the embedded URI for pg_restore -h.
- Pre-install vector + vectorscale extensions on the target before restore.
- Exclude timescaledb + timescaledb_toolkit from the dump (pgembed lacks them).
- Call request_explicit_shutdown() (sync) before await driver.shutdown().
- Verification uses the correct column name ``text`` (not ``content``).

Additional improvements:
- ``--dry-run`` flag: dump + count source rows + count target rows, no restore.
- Clear error messages on embedded start failure and on real pg_restore errors.
- PGPASSWORD passed via subprocess env; dump file size + restore duration printed.
- Post-restore verification: per-table row counts, random memory spot-check,
  and diskann index existence check on the target.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Tables to verify after migration (name -> expected to exist on source).
VERIFICATION_TABLES: tuple[str, ...] = (
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

# Extensions that pgembed does NOT ship — must be excluded from the dump so
# pg_restore does not fail with "extension ... is not available".
EXCLUDED_EXTENSIONS: tuple[str, ...] = ("timescaledb", "timescaledb_toolkit")

# Extensions that pgembed DOES ship and the dump will try to CREATE — we
# pre-install them so the CREATE EXTENSION statements in the dump become no-ops
# (pg_restore --clean --if-exists will still try to recreate them).
PREINSTALL_EXTENSIONS: tuple[str, ...] = ("vector", "vectorscale")


def _pgembed_bin_dir() -> Path:
    """Locate the bundled pgembed ``pginstall/bin`` directory.

    Falls back to PATH lookup if pgembed is not importable from this process
    (e.g. when run via ``python3 scripts/...`` outside the venv). In that
    fallback case we still prefer pgembed binaries if they exist at the
    conventional ``.venv/lib/pythonX.Y/site-packages/pgembed/pginstall/bin``
    path relative to the script location.
    """
    try:
        import pgembed  # type: ignore[import-not-found]
    except ImportError:
        pgembed_path: str | None = None
    else:
        pgembed_path = pgembed.__file__
    if pgembed_path is not None:
        # pgembed/__init__.py -> pgembed/pginstall/bin
        return Path(pgembed_path).resolve().parent / "pginstall" / "bin"
    # Fallback: conventional venv layout relative to this script.
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        Path(__file__).resolve().parent.parent
        / ".venv"
        / "lib"
        / py
        / "site-packages"
        / "pgembed"
        / "pginstall"
        / "bin"
    )


def _resolve_binary(name: str, prefer: str = "pgembed") -> str | Path:
    """Resolve a pg client binary.

    ``prefer`` controls which source is tried first:
    - ``"pgembed"`` (default): pgembed's bundled pg17 binary, then PATH.
      Used for ``pg_restore`` — the target is pg17, so matching the
      client major version avoids version-compatibility surprises.
    - ``"system"``: PATH first, then pgembed. Used for ``pg_dump`` —
      ``pg_dump`` must be >= the source server version, and the source
      is typically pg18 (timescaledb-ha:pg18) while pgembed ships pg17.
      A pg17 ``pg_dump`` aborts with "server version mismatch" against a
      pg18 source, so the system pg18 binary is the correct choice.
    """
    pgembed_bin = _pgembed_bin_dir() / name
    system_bin = shutil.which(name)

    candidates: list[str | Path]
    if prefer == "system":
        candidates = [system_bin, pgembed_bin] if system_bin else [pgembed_bin]
    else:
        candidates = [pgembed_bin, system_bin] if system_bin else [pgembed_bin]

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, Path) and not (c.exists() and os.access(c, os.X_OK)):
            continue
        return c

    print(
        f"ERROR: {name} not found (neither pgembed at {pgembed_bin} nor PATH).",
        file=sys.stderr,
    )
    raise FileNotFoundError(name)


def parse_db_url(url: str) -> dict[str, str]:
    """Parse postgresql:// URL into pg_dump/pg_restore connection params.

    Handles Unix socket URIs of the form
    ``postgresql://postgres:@/postgres?host=/path/to/data`` by extracting the
    ``?host=`` query parameter as the effective host (Bug 2 fix).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        # Unix socket URI: host lives in the ?host= query param.
        host_match = re.search(r"[?&]host=([^&]+)", url)
        host = host_match.group(1) if host_match else "localhost"
    return {
        "host": host,
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "postgres",
    }


def _run_pg_dump(
    source_url: str, dump_file: Path, pg_dump_bin: str | Path
) -> subprocess.CompletedProcess[str]:
    """Run pg_dump against the source, excluding extensions pgembed lacks."""
    src = parse_db_url(source_url)
    env = os.environ.copy()
    if src["password"]:
        env["PGPASSWORD"] = src["password"]
    cmd: list[str | Path] = [
        pg_dump_bin,
        "-h",
        src["host"],
        "-p",
        src["port"],
        "-U",
        src["user"],
        "-d",
        src["dbname"],
        "-Fc",
        "-f",
        str(dump_file),
        "--no-owner",
        "--no-privileges",
    ]
    for ext in EXCLUDED_EXTENSIONS:
        cmd.append(f"--exclude-extension={ext}")
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def _run_pg_restore(
    target_uri: str,
    dump_file: Path,
    pg_restore_bin: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run pg_restore into the embedded target via its Unix socket URI."""
    tgt = parse_db_url(target_uri)
    env = os.environ.copy()
    if tgt["password"]:
        env["PGPASSWORD"] = tgt["password"]
    return subprocess.run(
        [
            pg_restore_bin,
            "-h",
            tgt["host"],
            "-p",
            tgt["port"],
            "-U",
            tgt["user"],
            "-d",
            tgt["dbname"],
            "--no-owner",
            "--no-privileges",
            "--clean",
            "--if-exists",
            str(dump_file),
        ],
        env=env,
        capture_output=True,
        text=True,
    )


async def _start_embedded(target_dir: Path) -> tuple[str, object]:
    """Start (or attach to) the embedded pgembed server. Returns (uri, driver)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from memini_ai.postgres.driver import EmbeddedPGDriver  # type: ignore

    driver = EmbeddedPGDriver(target_dir)
    try:
        uri = await driver.get_uri()
    except (OSError, RuntimeError) as e:
        print(f"ERROR: failed to start embedded Postgres: {e}", file=sys.stderr)
        raise
    return uri, driver


async def _preinstall_extensions(target_uri: str) -> None:
    """Pre-install vector + vectorscale on the target before restore (Bug 3)."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(target_uri)
    try:
        for ext in PREINSTALL_EXTENSIONS:
            try:
                await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
                print(f"  Pre-installed extension: {ext}")
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: could not pre-install {ext}: {e}", file=sys.stderr)
    finally:
        await conn.close()


async def _count_rows(db_uri: str, table: str) -> int:
    """Return ``SELECT count(*) FROM <table>`` or -1 if table is missing."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(db_uri)
    try:
        try:
            return int(await conn.fetchval(f"SELECT count(*) FROM {table}"))
        except asyncpg.UndefinedTableError:
            return -1
    finally:
        await conn.close()


async def _count_source_rows(source_url: str) -> dict[str, int]:
    """Count rows in each verification table on the source DB."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(source_url)
    try:
        counts: dict[str, int] = {}
        for t in VERIFICATION_TABLES:
            try:
                counts[t] = int(await conn.fetchval(f"SELECT count(*) FROM {t}"))
            except asyncpg.UndefinedTableError:
                counts[t] = -1
        return counts
    finally:
        await conn.close()


async def _verify_migration(source_url: str, target_uri: str) -> bool:
    """Post-restore verification: row counts, spot-check, index check (Bug 6).

    Returns True if all checks pass, False otherwise. Prints a clear
    PASS/FAIL summary.
    """
    import asyncpg  # type: ignore[import-not-found]

    src_counts = await _count_source_rows(source_url)
    tgt_counts: dict[str, int] = {}
    conn = await asyncpg.connect(target_uri)
    try:
        for t in VERIFICATION_TABLES:
            try:
                tgt_counts[t] = int(await conn.fetchval(f"SELECT count(*) FROM {t}"))
            except asyncpg.UndefinedTableError:
                tgt_counts[t] = -1

        # Spot-check: pick a random memory and compare text + embedding.
        spot_ok = False
        spot_detail = ""
        try:
            row = await conn.fetchrow(
                "SELECT id, text, embedding FROM memories ORDER BY random() LIMIT 1"
            )
            if row is not None:
                mem_id = row["id"]
                src_conn = await asyncpg.connect(source_url)
                try:
                    src_row = await src_conn.fetchrow(
                        "SELECT text, embedding FROM memories WHERE id = $1",
                        mem_id,
                    )
                finally:
                    await src_conn.close()
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
        diskann_indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname='public' AND indexdef ILIKE '%USING diskann%' "
            "ORDER BY indexname"
        )
        diskann_names = [r["indexname"] for r in diskann_indexes]
    finally:
        await conn.close()

    # Build the summary.
    print("\n--- Verification ---")
    all_counts_ok = True
    for t in VERIFICATION_TABLES:
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


def _vectors_equal(a: object, b: object) -> bool:
    """Compare two pgvector values (list[float] or str) for equality."""
    if a == b:
        return True
    # pgvector may return a string like "[0.1,0.2,...]" — normalize.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate memini-ai data to embedded server"
    )
    parser.add_argument(
        "--from",
        dest="source_url",
        help="Source MEMINI_DB_URL (default: $MEMINI_DB_URL)",
    )
    parser.add_argument(
        "--to",
        dest="target_dir",
        help="Target data dir (default: $MEMINI_PGEMBED_DATA_DIR or "
        "~/.local/share/memini-ai/pgembed/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dump and count rows but do NOT restore into the embedded server.",
    )
    args = parser.parse_args()

    source_url = args.source_url or os.environ.get("MEMINI_DB_URL")
    if not source_url:
        print("ERROR: --from or $MEMINI_DB_URL required", file=sys.stderr)
        return 1

    target_dir = Path(
        args.target_dir
        or os.environ.get(
            "MEMINI_PGEMBED_DATA_DIR", "~/.local/share/memini-ai/pgembed/data"
        )
    ).expanduser()

    print(f"Source: {source_url}")
    print(f"Target: {target_dir}")
    print(f"Dry-run: {args.dry_run}")

    # Resolve client binaries.
    # pg_dump: prefer SYSTEM (must be >= source server version; source is
    #   typically pg18 while pgembed ships pg17, and pg17 pg_dump refuses
    #   to dump a pg18 server).
    # pg_restore: prefer PGEMBED (must match the pg17 target major version).
    try:
        pg_dump_bin = _resolve_binary("pg_dump", prefer="system")
        pg_restore_bin = _resolve_binary("pg_restore", prefer="pgembed")
    except FileNotFoundError:
        return 1
    print(f"  pg_dump:    {pg_dump_bin}")
    print(f"  pg_restore: {pg_restore_bin}")

    # Step 1: pg_dump source (exclude timescaledb extensions — Bug 4).
    print("\nStep 1/4: Dumping source database...")
    dump_file = target_dir.parent / f"memini-migrate-{os.getpid()}.dump"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = _run_pg_dump(source_url, dump_file, pg_dump_bin)
    dump_secs = time.monotonic() - t0
    if result.returncode != 0:
        print(f"pg_dump failed ({dump_secs:.1f}s):\n{result.stderr}", file=sys.stderr)
        dump_file.unlink(missing_ok=True)
        return 1
    dump_kb = dump_file.stat().st_size // 1024
    print(f"  Dumped to {dump_file} ({dump_kb} KB) in {dump_secs:.1f}s")

    # Step 2: Start embedded server (auto-initializes).
    print("\nStep 2/4: Starting embedded server...")
    try:
        uri, driver = asyncio.run(_start_embedded(target_dir))
    except (OSError, RuntimeError):
        dump_file.unlink(missing_ok=True)
        return 1
    print(f"  Embedded server started: {uri}")

    # Step 2b: Pre-install extensions on the target (Bug 3).
    print("\n  Pre-installing extensions on target...")
    asyncio.run(_preinstall_extensions(uri))

    # Source row counts (for both dry-run and real run).
    print("\n  Counting source rows...")
    src_counts = asyncio.run(_count_source_rows(source_url))
    for t in VERIFICATION_TABLES:
        print(f"    source {t}: {src_counts.get(t, -1)}")

    # Target row counts after pre-install (before restore).
    print("\n  Counting target rows (post pre-install, pre-restore)...")
    for t in VERIFICATION_TABLES:
        c = asyncio.run(_count_rows(uri, t))
        print(f"    target {t}: {c}")

    if args.dry_run:
        print("\n--dry-run: skipping restore. Dump file left at:", dump_file)

        # Stop the embedded server we just started.
        async def stop_dry() -> None:
            # request_explicit_shutdown is SYNC (Bug 5) — do not await it.
            driver.request_explicit_shutdown()
            await driver.shutdown()

        asyncio.run(stop_dry())
        print("Dry-run complete. No data was restored.")
        return 0

    # Step 3: pg_restore into embedded (via Unix socket — Bug 2).
    print("\nStep 3/4: Restoring into embedded server...")
    t0 = time.monotonic()
    result = _run_pg_restore(uri, dump_file, pg_restore_bin)
    restore_secs = time.monotonic() - t0
    print(f"  pg_restore finished in {restore_secs:.1f}s (rc={result.returncode})")

    # Cleanup dump file.
    dump_file.unlink(missing_ok=True)

    # pg_restore returns nonzero for harmless warnings (e.g. role does not
    # exist). Only fail on real ERROR: lines that are NOT timescaledb-related
    # (which we excluded but pg_restore may still complain about).
    if result.returncode != 0:
        real_errors = [
            line
            for line in result.stderr.splitlines()
            if "ERROR:" in line
            and "timescaledb" not in line.lower()
            and "extension" not in line.lower()
        ]
        if real_errors:
            print(
                "pg_restore had real errors:\n" + "\n".join(real_errors),
                file=sys.stderr,
            )
            print(f"\nFull stderr:\n{result.stderr}", file=sys.stderr)

            # Still try to stop the server before bailing.
            async def stop_err() -> None:
                driver.request_explicit_shutdown()
                await driver.shutdown()

            asyncio.run(stop_err())
            return 1
        print(f"  Restored with warnings (first 500 chars): {result.stderr[:500]}")
    else:
        print("  Restored cleanly.")

    # Step 4: Verification (Bug 6 — uses correct column name ``text``).
    print("\nStep 4/4: Verifying migration...")
    verified = asyncio.run(_verify_migration(source_url, uri))

    # Stop the embedded server (Bug 5: request_explicit_shutdown is sync).
    async def stop() -> None:
        driver.request_explicit_shutdown()
        await driver.shutdown()

    asyncio.run(stop())

    if verified:
        print("\n✅ Migration complete!")
        return 0
    print(
        "\n⚠️  Migration finished but verification FAILED — see above.", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
