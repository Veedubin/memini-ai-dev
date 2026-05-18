"""File tracker for SQLite-based file tracking and persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass
class TrackedFile:
    """Represents a tracked file with its metadata."""

    path: str
    content_hash: str
    size: int
    modified: datetime
    indexed_at: datetime


@dataclass
class FileTracker:
    """SQLite-based file tracker for persistence of indexed files.

    Tracks files with their content hashes to detect changes and
    support incremental re-indexing.
    """

    _db_path: str
    _conn: aiosqlite.Connection | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __init__(self, db_path: str | Path) -> None:
        """Initialize file tracker.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = str(db_path)
        self._lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database connection and create tables."""
        async with self._lock:
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._create_tables()

    async def _create_tables(self) -> None:
        """Create the tracked_files table if it doesn't exist."""
        conn = self._conn
        if conn is None:
            raise RuntimeError("FileTracker not initialized")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tracked_files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_hash
            ON tracked_files(content_hash)
        """)
        await conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def upsert_file(self, file: TrackedFile) -> None:
        """Insert or update a tracked file.

        Args:
            file: The file to track.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            await conn.execute(
                """
                INSERT INTO tracked_files (path, content_hash, size, modified, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    size = excluded.size,
                    modified = excluded.modified,
                    indexed_at = excluded.indexed_at
                """,
                (
                    file.path,
                    file.content_hash,
                    file.size,
                    file.modified.isoformat(),
                    file.indexed_at.isoformat(),
                ),
            )
            await conn.commit()

    async def get_file(self, path: str) -> TrackedFile | None:
        """Get tracked file by path.

        Args:
            path: The file path to look up.

        Returns:
            TrackedFile if found, None otherwise.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            async with conn.execute(
                "SELECT path, content_hash, size, modified, indexed_at FROM tracked_files WHERE path = ?",
                (path,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return TrackedFile(
                    path=row[0],
                    content_hash=row[1],
                    size=row[2],
                    modified=datetime.fromisoformat(row[3]),
                    indexed_at=datetime.fromisoformat(row[4]),
                )

    async def get_file_hash(self, path: str) -> str | None:
        """Get the content hash for a file path.

        Args:
            path: The file path to look up.

        Returns:
            Content hash string if found, None otherwise.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            async with conn.execute(
                "SELECT content_hash FROM tracked_files WHERE path = ?",
                (path,),
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

    async def remove_file(self, path: str) -> None:
        """Remove a file from tracking.

        Args:
            path: The file path to remove.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            await conn.execute("DELETE FROM tracked_files WHERE path = ?", (path,))
            await conn.commit()

    async def get_all_paths(self) -> list[str]:
        """Get all tracked file paths.

        Returns:
            List of all tracked file paths.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            async with conn.execute("SELECT path FROM tracked_files") as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """Get tracking statistics.

        Returns:
            Dictionary with file count and total size.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            async with conn.execute(
                "SELECT COUNT(*), SUM(size) FROM tracked_files"
            ) as cursor:
                row = await cursor.fetchone()
                if row is None or row[0] is None:
                    return {"file_count": 0, "total_size": 0}
                return {"file_count": row[0], "total_size": row[1] or 0}

    async def clear(self) -> None:
        """Clear all tracked files."""
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            await conn.execute("DELETE FROM tracked_files")
            await conn.commit()
