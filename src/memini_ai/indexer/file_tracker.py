"""File tracker for SQLite-based file tracking and persistence."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
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
        """Create the tracked_files + project_chunks tables if missing."""
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS project_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                content TEXT NOT NULL,
                language TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                UNIQUE(path, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_chunks_path
            ON project_chunks(path)
        """)
        await conn.commit()

    async def insert_chunks(self, chunks: Sequence[Any]) -> int:
        """Insert or update a batch of chunks in project_chunks.

        Accepts objects with ``path``, ``chunk_index``, ``total_chunks``,
        ``content``, ``language``, ``start_line`` and ``end_line``
        attributes (duck-typed so the chunker stays decoupled).

        Args:
            chunks: Iterable of chunk-like objects to persist.

        Returns:
            Number of rows written.

        Raises:
            RuntimeError: If the tracker is not initialized.
        """
        rows = [
            (
                c.path,
                c.chunk_index,
                c.total_chunks,
                c.content,
                c.language,
                c.start_line,
                c.end_line,
            )
            for c in chunks
        ]
        if not rows:
            return 0
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            await conn.executemany(
                """
                INSERT INTO project_chunks (
                    path, chunk_index, total_chunks, content,
                    language, start_line, end_line
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path, chunk_index) DO UPDATE SET
                    total_chunks = excluded.total_chunks,
                    content = excluded.content,
                    language = excluded.language,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line
                """,
                rows,
            )
            await conn.commit()
        return len(rows)

    async def count_chunks(self) -> int:
        """Count persisted chunks in project_chunks.

        Returns:
            Total number of chunk rows.

        Raises:
            RuntimeError: If the tracker is not initialized.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            async with conn.execute("SELECT COUNT(*) FROM project_chunks") as cursor:
                row = await cursor.fetchone()
                return int(row[0]) if row else 0

    async def get_chunks_for_file(self, path: str) -> list[dict[str, Any]]:
        """Get all persisted chunks for a file, ordered by chunk_index.

        Args:
            path: File path to look up.

        Returns:
            List of chunk row dicts ordered by chunk_index.

        Raises:
            RuntimeError: If the tracker is not initialized.
        """
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                """
                SELECT path, chunk_index, total_chunks, content,
                       language, start_line, end_line
                FROM project_chunks WHERE path = ? ORDER BY chunk_index
                """,
                (path,),
            ) as cursor:
                rows = await cursor.fetchall()
            return [dict(r) for r in rows]

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
        """Clear all tracked files and persisted chunks."""
        async with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("FileTracker not initialized")
            await conn.execute("DELETE FROM project_chunks")
            await conn.execute("DELETE FROM tracked_files")
            await conn.commit()
