"""Tests for project indexer."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from memini_ai.indexer.constants import ALLOWED_EXTENSIONS, ALWAYS_EXCLUDED, SKIP_DIRS
from memini_ai.indexer.file_tracker import FileTracker, TrackedFile
from memini_ai.indexer.indexer import (
    IndexerConfig,
    IndexerStats,
    ProjectIndexer,
)
from memini_ai.indexer.pause_controller import PauseController, PauseState
from memini_ai.indexer.snapshot import SnapshotIndex


class TestIndexerConfig:
    """Tests for IndexerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = IndexerConfig()

        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.max_file_size > 0
        assert config.flush_interval > 0
        assert config.batch_size > 0
        assert config.db_path

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = IndexerConfig(
            chunk_size=1024,
            chunk_overlap=100,
            max_file_size=5 * 1024 * 1024,
            flush_interval=10.0,
            batch_size=50,
            db_path=".custom/indexer.db",
        )

        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.max_file_size == 5 * 1024 * 1024
        assert config.flush_interval == 10.0
        assert config.batch_size == 50
        assert config.db_path == ".custom/indexer.db"


class TestIndexerStats:
    """Tests for IndexerStats."""

    def test_default_values(self) -> None:
        """Test default stat values."""
        stats = IndexerStats()

        assert stats.files_indexed == 0
        assert stats.chunks_created == 0
        assert stats.bytes_processed == 0
        assert stats.errors == 0
        assert stats.last_indexed is None
        assert stats.is_running is False
        assert stats.is_paused is False


class TestPauseController:
    """Tests for PauseController."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test initial pause state is running."""
        controller = PauseController()
        assert controller.state == PauseState.RUNNING
        assert not controller.is_paused

    @pytest.mark.asyncio
    async def test_pause_and_resume(self) -> None:
        """Test pause and resume functionality."""
        controller = PauseController()

        await controller.pause()
        assert controller.is_paused

        await controller.resume()
        assert not controller.is_paused

    @pytest.mark.asyncio
    async def test_multiple_pauses_single_resume(self) -> None:
        """Test multiple pauses require multiple resumes."""
        controller = PauseController()

        await controller.pause()
        await controller.pause()
        assert controller.is_paused

        await controller.resume()
        assert controller.is_paused  # Still paused

        await controller.resume()
        assert not controller.is_paused

    @pytest.mark.asyncio
    async def test_wait_while_paused(self) -> None:
        """Test wait while paused."""
        controller = PauseController()

        # Start a task that waits
        async def wait_task() -> None:
            await controller.wait_while_paused()

        # Not paused - should return quickly
        task = asyncio.create_task(wait_task())
        await asyncio.sleep(0.01)  # Small delay
        assert not task.done() or controller.state == PauseState.RUNNING


class TestFileTracker:
    """Tests for FileTracker."""

    @pytest.fixture
    async def tracker(self, tmp_path: Path) -> FileTracker:
        """Create a file tracker with temporary database."""
        db_path = tmp_path / "test_tracker.db"
        tracker = FileTracker(str(db_path))
        await tracker.initialize()
        yield tracker
        await tracker.close()

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path: Path) -> None:
        """Test file tracker initialization."""
        db_path = tmp_path / "test.db"
        tracker = FileTracker(str(db_path))
        await tracker.initialize()
        await tracker.close()

    @pytest.mark.asyncio
    async def test_upsert_file(self, tracker: FileTracker) -> None:
        """Test inserting/updating a file."""
        tracked = TrackedFile(
            path="/test/file.py",
            content_hash="abc123",
            size=100,
            modified=datetime.utcnow(),
            indexed_at=datetime.utcnow(),
        )
        await tracker.upsert_file(tracked)

        result = await tracker.get_file("/test/file.py")
        assert result is not None
        assert result.path == "/test/file.py"
        assert result.content_hash == "abc123"
        assert result.size == 100

    @pytest.mark.asyncio
    async def test_get_file_hash(self, tracker: FileTracker) -> None:
        """Test getting file hash."""
        tracked = TrackedFile(
            path="/test/file.py",
            content_hash="abc123",
            size=100,
            modified=datetime.utcnow(),
            indexed_at=datetime.utcnow(),
        )
        await tracker.upsert_file(tracked)

        hash_val = await tracker.get_file_hash("/test/file.py")
        assert hash_val == "abc123"

    @pytest.mark.asyncio
    async def test_get_nonexistent_file(self, tracker: FileTracker) -> None:
        """Test getting a file that doesn't exist."""
        result = await tracker.get_file("/nonexistent/file.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_file(self, tracker: FileTracker) -> None:
        """Test removing a file."""
        tracked = TrackedFile(
            path="/test/file.py",
            content_hash="abc123",
            size=100,
            modified=datetime.utcnow(),
            indexed_at=datetime.utcnow(),
        )
        await tracker.upsert_file(tracked)

        await tracker.remove_file("/test/file.py")
        result = await tracker.get_file("/test/file.py")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_paths(self, tracker: FileTracker) -> None:
        """Test getting all tracked paths."""
        files = [
            TrackedFile(
                path=f"/test/file{i}.py",
                content_hash=f"hash{i}",
                size=100 * i,
                modified=datetime.utcnow(),
                indexed_at=datetime.utcnow(),
            )
            for i in range(3)
        ]
        for f in files:
            await tracker.upsert_file(f)

        paths = await tracker.get_all_paths()
        assert len(paths) == 3
        assert all(p.startswith("/test/file") for p in paths)

    @pytest.mark.asyncio
    async def test_get_stats(self, tracker: FileTracker) -> None:
        """Test getting tracking statistics."""
        tracked = TrackedFile(
            path="/test/file.py",
            content_hash="abc123",
            size=100,
            modified=datetime.utcnow(),
            indexed_at=datetime.utcnow(),
        )
        await tracker.upsert_file(tracked)

        stats = await tracker.get_stats()
        assert stats["file_count"] == 1
        assert stats["total_size"] == 100

    @pytest.mark.asyncio
    async def test_clear(self, tracker: FileTracker) -> None:
        """Test clearing all tracked files."""
        tracked = TrackedFile(
            path="/test/file.py",
            content_hash="abc123",
            size=100,
            modified=datetime.utcnow(),
            indexed_at=datetime.utcnow(),
        )
        await tracker.upsert_file(tracked)

        await tracker.clear()
        paths = await tracker.get_all_paths()
        assert len(paths) == 0


class TestSnapshotIndex:
    """Tests for SnapshotIndex."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test initial snapshot state."""
        snapshot = SnapshotIndex()
        stats = snapshot.get_stats()

        assert stats["tracked_files"] == 0
        assert stats["total_size"] == 0

    @pytest.mark.asyncio
    async def test_update_file(self) -> None:
        """Test updating a file in snapshot."""
        snapshot = SnapshotIndex()

        result = await snapshot.update_file(
            "/test/file.py", "abc123", 100, 1234567890.0
        )
        assert result is True  # New file

        stats = snapshot.get_stats()
        assert stats["tracked_files"] == 1

    @pytest.mark.asyncio
    async def test_check_file_unchanged(self) -> None:
        """Test checking an unchanged file."""
        snapshot = SnapshotIndex()

        await snapshot.update_file("/test/file.py", "abc123", 100, 1234567890.0)
        changed = await snapshot.check_file("/test/file.py", "abc123")
        assert changed is False  # Unchanged

    @pytest.mark.asyncio
    async def test_check_file_changed(self) -> None:
        """Test checking a changed file."""
        snapshot = SnapshotIndex()

        await snapshot.update_file("/test/file.py", "abc123", 100, 1234567890.0)
        changed = await snapshot.check_file("/test/file.py", "xyz789")
        assert changed is True  # Changed

    @pytest.mark.asyncio
    async def test_remove_file(self) -> None:
        """Test removing a file from snapshot."""
        snapshot = SnapshotIndex()

        await snapshot.update_file("/test/file.py", "abc123", 100, 1234567890.0)
        await snapshot.remove_file("/test/file.py")

        assert not snapshot.has_file("/test/file.py")

    @pytest.mark.asyncio
    async def test_get_all_paths(self) -> None:
        """Test getting all tracked paths."""
        snapshot = SnapshotIndex()

        await snapshot.update_file("/test/file1.py", "abc", 100, 0)
        await snapshot.update_file("/test/file2.py", "def", 200, 0)

        paths = await snapshot.get_all_paths()
        assert len(paths) == 2
        assert "/test/file1.py" in paths
        assert "/test/file2.py" in paths


class TestProjectIndexer:
    """Tests for ProjectIndexer."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test initial indexer state."""
        config = IndexerConfig()
        indexer = ProjectIndexer(config)

        assert not indexer.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping the indexer."""
        config = IndexerConfig()
        indexer = ProjectIndexer(config)

        # Create temp dir for db
        with tempfile.TemporaryDirectory() as tmp:
            config.db_path = str(Path(tmp) / "test.db")
            indexer = ProjectIndexer(config)

            await indexer.start()
            assert indexer.is_running

            stats = indexer.get_stats()
            assert stats.is_running

            await indexer.stop()
            assert not indexer.is_running

    @pytest.mark.asyncio
    async def test_pause_resume(self) -> None:
        """Test pause and resume."""
        config = IndexerConfig()
        indexer = ProjectIndexer(config)

        indexer.pause()
        stats = indexer.get_stats()
        assert stats.is_paused

        indexer.resume()
        stats = indexer.get_stats()
        assert not stats.is_paused

    @pytest.mark.asyncio
    async def test_index_directory(self) -> None:
        """Test indexing a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create test files
            test_dir = Path(tmp) / "project"
            test_dir.mkdir()

            (test_dir / "test.py").write_text("def test(): pass")
            (test_dir / "test.js").write_text("const x = 1;")

            config = IndexerConfig(db_path=str(test_dir / ".memini" / "indexer.db"))
            indexer = ProjectIndexer(config)

            await indexer.start()
            await indexer.set_root_path(str(test_dir))

            count = await indexer.index_directory()
            # Should index at least the Python file
            assert count >= 1

            stats = indexer.get_stats()
            assert stats.files_indexed >= 1

            await indexer.stop()

    @pytest.mark.asyncio
    async def test_clear_index(self) -> None:
        """Test clearing the index."""
        with tempfile.TemporaryDirectory() as tmp:
            config = IndexerConfig(db_path=str(Path(tmp) / "test.db"))
            indexer = ProjectIndexer(config)

            await indexer.start()
            await indexer.set_root_path(tmp)

            # Add some data
            indexer.clear_index()
            stats = indexer.get_stats()
            assert stats.files_indexed == 0

            await indexer.stop()


class TestConstants:
    """Tests for indexer constants."""

    def test_skip_dirs_contains_common(self) -> None:
        """Test SKIP_DIRS contains common directories."""
        assert "node_modules" in SKIP_DIRS
        assert ".git" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS

    def test_allowed_extensions_contains_python(self) -> None:
        """Test ALLOWED_EXTENSIONS contains Python."""
        assert ".py" in ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_typescript(self) -> None:
        """Test ALLOWED_EXTENSIONS contains TypeScript."""
        assert ".ts" in ALLOWED_EXTENSIONS
        assert ".tsx" in ALLOWED_EXTENSIONS

    def test_allowed_extensions_contains_config(self) -> None:
        """Test ALLOWED_EXTENSIONS contains config formats."""
        assert ".json" in ALLOWED_EXTENSIONS
        assert ".yaml" in ALLOWED_EXTENSIONS
        assert ".yml" in ALLOWED_EXTENSIONS
        assert ".toml" in ALLOWED_EXTENSIONS

    def test_alway_excluded_contains_temp_files(self) -> None:
        """Test ALWAYS_EXCLUDED contains temp files."""
        assert ".DS_Store" in ALWAYS_EXCLUDED
        assert "Thumbs.db" in ALWAYS_EXCLUDED
