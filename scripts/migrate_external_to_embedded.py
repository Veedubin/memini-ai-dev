#!/usr/bin/env python3
"""Migrate memini-ai data from an external Postgres to the embedded pgembed server.

Usage:
    memini-ai migrate --from=postgresql://user:pass@host:port/db
    memini-ai migrate  # uses $MEMINI_DB_URL as source

This uses pg_dump + pg_restore under the hood. The embedded pgembed server
must NOT be running before invoking this script (otherwise pg_restore will
fail with "database is being accessed by other users").
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def parse_db_url(url: str) -> dict:
    """Parse postgresql:// URL into pg_dump connection params."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "postgres",
    }


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
        help="Target data dir (default: $MEMINI_PGEMBED_DATA_DIR or ~/.local/share/memini-ai/pgembed/data)",
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

    # Check pg_dump / pg_restore available
    for tool in ("pg_dump", "pg_restore"):
        if not shutil.which(tool):
            print(
                f"ERROR: {tool} not found in PATH. Install postgresql-client.",
                file=sys.stderr,
            )
            return 1

    # Step 1: pg_dump source
    print("Step 1/3: Dumping source database...")
    dump_file = target_dir.parent / f"memini-migrate-{os.getpid()}.dump"
    src_params = parse_db_url(source_url)
    env = os.environ.copy()
    if src_params["password"]:
        env["PGPASSWORD"] = src_params["password"]

    result = subprocess.run(
        [
            "pg_dump",
            "-h",
            src_params["host"],
            "-p",
            src_params["port"],
            "-U",
            src_params["user"],
            "-d",
            src_params["dbname"],
            "-Fc",  # custom format (compressed, supports pg_restore)
            "-f",
            str(dump_file),
            "--no-owner",  # don't set ownership
            "--no-privileges",  # don't grant/revoke
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"pg_dump failed: {result.stderr}", file=sys.stderr)
        return 1

    print(f"  Dumped to {dump_file} ({dump_file.stat().st_size // 1024} KB)")

    # Step 2: Start embedded server (auto-initializes)
    print("Step 2/3: Starting embedded server...")

    async def start_embedded():
        # Add src to path so we can import memini_ai
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from memini_ai.postgres.driver import EmbeddedPGDriver

        driver = EmbeddedPGDriver(target_dir)
        uri = await driver.get_uri()
        return uri, driver

    uri, driver = asyncio.run(start_embedded())
    print(f"  Embedded server started: {uri}")

    # Step 3: pg_restore into embedded
    print("Step 3/3: Restoring into embedded server...")
    # Parse the URI to get connection params for pg_restore
    restore_params = parse_db_url(uri)
    env2 = os.environ.copy()
    if restore_params["password"]:
        env2["PGPASSWORD"] = restore_params["password"]

    result = subprocess.run(
        [
            "pg_restore",
            "-h",
            restore_params["host"],
            "-p",
            restore_params["port"],
            "-U",
            restore_params["user"],
            "-d",
            restore_params["dbname"],
            "--no-owner",
            "--no-privileges",
            "--clean",  # drop objects before creating
            "--if-exists",  # don't error if objects don't exist
            str(dump_file),
        ],
        env=env2,
        capture_output=True,
        text=True,
    )

    # Cleanup dump file
    dump_file.unlink(missing_ok=True)

    # Stop embedded server
    async def stop():
        await driver.shutdown()

    asyncio.run(stop())

    if result.returncode != 0:
        # pg_restore often returns nonzero for harmless errors (e.g., role already exists)
        # Only fail on stderr containing "ERROR:"
        if "ERROR:" in result.stderr:
            print(f"pg_restore had errors:\n{result.stderr}", file=sys.stderr)
            return 1
        print(f"  Restored (with warnings: {result.stderr[:500]})")

    print("✅ Migration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
