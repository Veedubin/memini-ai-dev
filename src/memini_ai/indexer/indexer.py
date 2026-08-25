"""Project indexer orchestrator - main indexing logic and coordination."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memini_ai.indexer.chunker import Chunk, SemanticChunker
from memini_ai.indexer.constants import (
    ALLOWED_EXTENSIONS,
    ALWAYS_EXCLUDED,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_FILE_SIZE,
    SKIP_DIRS,
)
from memini_ai.indexer.file_tracker import FileTracker, TrackedFile
from memini_ai.indexer.pause_controller import PauseController, PauseState
from memini_ai.indexer.snapshot import SnapshotIndex
from memini_ai.indexer.watcher import FileWatcher
from memini_ai.utils.hash import hash_file

logger = logging.getLogger(__name__)

# Glob-syntax entries in SKIP_DIRS / ALWAYS_EXCLUDED (e.g. "*.egg-info",
# "~$*", "*.swp") never match a plain membership test — they need fnmatch.
_SKIP_DIR_PATTERNS = tuple(p for p in SKIP_DIRS if any(c in p for c in "*?["))
_EXACT_SKIP_DIRS = frozenset(SKIP_DIRS - set(_SKIP_DIR_PATTERNS))
_EXCLUDED_NAME_PATTERNS = tuple(
    p for p in ALWAYS_EXCLUDED if any(c in p for c in "*?[")
)
_EXACT_EXCLUDED_NAMES = frozenset(ALWAYS_EXCLUDED - set(_EXCLUDED_NAME_PATTERNS))


def _dir_excluded(name: str) -> bool:
    """Check whether a directory name is excluded (exact or glob match)."""
    return name in _EXACT_SKIP_DIRS or any(
        fnmatch.fnmatch(name, pat) for pat in _SKIP_DIR_PATTERNS
    )


def _file_excluded(name: str) -> bool:
    """Check whether a file name is always-excluded (exact or glob match)."""
    return name in _EXACT_EXCLUDED_NAMES or any(
        fnmatch.fnmatch(name, pat) for pat in _EXCLUDED_NAME_PATTERNS
    )


@dataclass
class IndexerConfig:
    """Configuration for the project indexer."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    flush_interval: float = 5.0  # seconds
    batch_size: int = 100
    db_path: str = ".memini/indexer.db"


@dataclass
class IndexerStats:
    """Statistics for the project indexer."""

    files_indexed: int = 0
    chunks_created: int = 0
    bytes_processed: int = 0
    errors: int = 0
    last_indexed: datetime | None = None
    is_running: bool = False
    is_paused: bool = False


@dataclass
class ChunkResult:
    """Result for a chunk from search."""

    path: str
    content: str
    chunk_index: int
    total_chunks: int
    start_line: int
    end_line: int
    score: float | None = None


@dataclass
class FileContentsResult:
    """Result for file contents reconstruction."""

    path: str
    content: str
    total_chunks: int


@dataclass
class SearchResult:
    """Result from a search operation."""

    path: str
    content: str
    score: float | None = None
    chunk_index: int = 0
    total_chunks: int = 0


class ProjectIndexer:
    """Main project indexer orchestrator.

    Coordinates file tracking, snapshot management, chunking,
    file watching, and batch flushing for efficient indexing.
    """

    _config: IndexerConfig
    _chunk_buffer: list[Chunk]
    _buffer_lock: asyncio.Lock
    _file_tracker: FileTracker
    _snapshot: SnapshotIndex
    _pause: PauseController
    _watcher: FileWatcher | None
    _chunker: SemanticChunker
    _root_path: str | None
    _stats: IndexerStats
    _running: bool
    _flush_task: asyncio.Task[None] | None

    def __init__(self, config: IndexerConfig) -> None:
        """Initialize project indexer.

        Args:
            config: Indexer configuration.
        """
        self._config = config
        self._chunk_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._file_tracker = FileTracker(config.db_path)
        self._snapshot = SnapshotIndex()
        self._pause = PauseController()
        self._chunker = SemanticChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self._root_path = None
        self._stats = IndexerStats()
        self._running = False
        self._flush_task = None
        self._watcher = None

    @property
    def is_running(self) -> bool:
        """Check if indexer is running."""
        return self._running

    def get_stats(self) -> IndexerStats:
        """Get current indexer statistics."""
        return self._stats

    def pause(self) -> None:
        """Pause indexing operations."""
        self._pause._state = PauseState.PAUSED
        self._stats.is_paused = True

    def resume(self) -> None:
        """Resume indexing operations."""
        self._pause._state = PauseState.RUNNING
        self._stats.is_paused = False

    async def set_root_path(self, path: str) -> None:
        """Set the root path for indexing.

        Args:
            path: Root directory path.
        """
        self._root_path = str(Path(path).resolve())
        await self._snapshot.set_root_path(self._root_path)

    async def start(self) -> None:
        """Start the project indexer."""
        if self._running:
            return

        # Ensure db path directory exists
        db_path = Path(self._config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file tracker
        await self._file_tracker.initialize()

        # Start flush task
        self._running = True
        self._stats.is_running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Start file watcher if root path set
        if self._root_path:
            await self._watcher_setup()

    async def _watcher_setup(self) -> None:
        """Set up the file watcher."""
        if not self._root_path:
            return

        watcher_cb = _WatcherCallback(self)
        self._watcher = FileWatcher(self._root_path, watcher_cb)
        await self._watcher.start()

    async def stop(self) -> None:
        """Stop the project indexer."""
        if not self._running:
            return

        self._running = False
        self._stats.is_running = False

        # Stop watcher
        if self._watcher:
            await self._watcher.stop()
            self._watcher = None

        # Flush remaining chunks
        await self._flush_chunks()

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        # Close file tracker
        await self._file_tracker.close()

    async def _flush_loop(self) -> None:
        """Background loop for periodic chunk flushing."""
        while self._running:
            await asyncio.sleep(self._config.flush_interval)
            if self._running and not self._stats.is_paused:
                await self._flush_chunks()

    async def _flush_chunks(self) -> None:
        """Flush buffered chunks to persistent storage (project_chunks).

        Chunks are persisted incrementally in batches so ``project_chunks``
        grows DURING long index runs (T-IDX-WEDGE-001). If the file tracker
        is not initialized (unit tests, tracker never started), falls back
        to stat-counting only. On a persistence failure the batch is put
        back at the front of the buffer for the next flush attempt.
        """
        async with self._buffer_lock:
            if not self._chunk_buffer:
                return

            chunks = self._chunk_buffer
            self._chunk_buffer = []

        try:
            written = await self._file_tracker.insert_chunks(chunks)
            self._stats.chunks_created += written
        except RuntimeError:
            # Tracker not initialized — preserve legacy count-only behavior.
            self._stats.chunks_created += len(chunks)
        except Exception:
            # Persistence failed: re-buffer for retry on the next flush.
            self._stats.errors += 1
            logger.exception("chunk_flush_failed", extra={"batch_size": len(chunks)})
            async with self._buffer_lock:
                self._chunk_buffer[:0] = chunks

    async def _index_file(self, path: str) -> bool:
        """Index a single file.

        Args:
            path: File path to index.

        Returns:
            True if file was indexed, False if skipped.
        """
        # Check pause state
        await self._pause.wait_while_paused()
        if not self._running:
            return False

        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return False

        # Check file size
        try:
            size = file_path.stat().st_size
            if size > self._config.max_file_size:
                return False
        except OSError:
            return False

        # Skip excluded paths (dir parts exact-match; file names may glob)
        parts = file_path.parts
        for part in parts:
            if part in _EXACT_SKIP_DIRS:
                return False
        if _file_excluded(file_path.name):
            return False

        # Check extension
        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False

        try:
            # Hash file (threaded: hashing large files is CPU/IO-bound)
            content_hash = await asyncio.to_thread(hash_file, path)
            size = file_path.stat().st_size
            modified = file_path.stat().st_mtime

            # Check if changed
            if not await self._snapshot.check_file(path, content_hash):
                return False

            # Read + chunk OFF the event loop (T-IDX-WEDGE-001): chunking is
            # pure-Python CPU work; running it inline starves every other
            # tool call for the whole duration of a large index run.
            chunks = await asyncio.to_thread(self._read_and_chunk, path)

            # Update snapshot
            await self._snapshot.update_file(path, content_hash, size, modified)

            # Buffer chunks
            async with self._buffer_lock:
                self._chunk_buffer.extend(chunks)

            # Update file tracker
            tracked = TrackedFile(
                path=path,
                content_hash=content_hash,
                size=size,
                modified=datetime.fromtimestamp(modified),
                indexed_at=datetime.utcnow(),
            )
            await self._file_tracker.upsert_file(tracked)

            self._stats.files_indexed += 1
            self._stats.bytes_processed += size
            self._stats.last_indexed = datetime.utcnow()

            return True

        except Exception:
            self._stats.errors += 1
            return False

    def _read_and_chunk(self, path: str) -> list[Chunk]:
        """Read a file and chunk its content (blocking; run in a thread).

        Args:
            path: File path to read.

        Returns:
            List of chunks for the file content.
        """
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        return self._chunker.chunk_file(content, path)

    async def index_directory(self, path: str | None = None) -> int:
        """Index all files in a directory.

        Args:
            path: Directory path. Uses root_path if not provided.

        Returns:
            Number of files indexed.
        """
        target = Path(path or self._root_path or ".")
        if not target.is_dir():
            return 0

        count = 0
        async for file_path in self._walk_directory(target):
            if await self._index_file(str(file_path)):
                count += 1
            # Fairness: guarantee the event loop a scheduling point between
            # files even when every per-file segment happens to be awaited
            # fast, so concurrent tool calls never queue behind a long run.
            await asyncio.sleep(0)

        return count

    async def _walk_directory(self, root: Path) -> AsyncGenerator[Path, None]:
        """Walk directory yielding file paths.

        Excluded directories (SKIP_DIRS, exact or glob names) are PRUNED —
        never descended into — so huge irrelevant trees (.venv, node_modules,
        build caches) cost zero traversal time.

        Args:
            root: Root directory.

        Yields:
            File paths.
        """

        def _iter(dir_path: Path) -> Iterator[Path]:
            try:
                entries = list(dir_path.iterdir())
            except OSError:
                return
            for entry in entries:
                if entry.is_dir():
                    if not _dir_excluded(entry.name):
                        yield from _iter(entry)
                elif entry.is_file() and not _file_excluded(entry.name):
                    yield entry

        for file_path in _iter(root):
            yield file_path

    async def search(self, query: str, options: Any) -> list[SearchResult]:
        """Search indexed files for matching chunks.

        Args:
            query: Search query.
            options: Search options.

        Returns:
            List of search results.
        """
        # This would integrate with memory search
        # For now, return empty list - full integration later
        return []

    async def get_file_contents(self, file_path: str) -> FileContentsResult | None:
        """Reconstruct file contents from indexed chunks.

        Args:
            file_path: Path to the file.

        Returns:
            FileContentsResult if found, None otherwise.
        """
        # This would retrieve chunks from storage
        # For now, return None - full integration later
        return None

    def clear_index(self) -> None:
        """Clear all indexed data."""
        self._snapshot._snapshots.clear()
        self._chunk_buffer.clear()
        asyncio.create_task(self._file_tracker.clear())
        self._stats = IndexerStats()


class _WatcherCallback:
    """Callback for file watcher events."""

    def __init__(self, indexer: ProjectIndexer) -> None:
        self._indexer = indexer

    async def on_file_created(self, path: str) -> None:
        """Handle file created event."""
        await self._indexer._index_file(path)

    async def on_file_modified(self, path: str) -> None:
        """Handle file modified event."""
        await self._indexer._index_file(path)

    async def on_file_deleted(self, path: str) -> None:
        """Handle file deleted event."""
        await self._indexer._snapshot.remove_file(path)
        await self._indexer._file_tracker.remove_file(path)

    async def on_file_moved(self, old_path: str, new_path: str) -> None:
        """Handle file moved event."""
        # Remove old and index new
        await self._indexer._snapshot.remove_file(old_path)
        await self._indexer._file_tracker.remove_file(old_path)
        await self._indexer._index_file(new_path)
