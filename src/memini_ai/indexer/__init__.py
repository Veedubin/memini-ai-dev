"""Indexer package - Project indexing with file tracking and chunking."""

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
from memini_ai.indexer.indexer import (
    ChunkResult,
    FileContentsResult,
    IndexerConfig,
    IndexerStats,
    ProjectIndexer,
)
from memini_ai.indexer.pause_controller import PauseController, PauseState
from memini_ai.indexer.snapshot import SnapshotIndex
from memini_ai.indexer.watcher import FileWatcher

__all__ = [
    # Constants
    "ALLOWED_EXTENSIONS",
    "ALWAYS_EXCLUDED",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MAX_FILE_SIZE",
    "SKIP_DIRS",
    # Classes
    "Chunk",
    "FileTracker",
    "FileWatcher",
    "IndexerConfig",
    "IndexerStats",
    "PauseController",
    "PauseState",
    "ProjectIndexer",
    "SemanticChunker",
    "SnapshotIndex",
    "TrackedFile",
    # Data classes
    "ChunkResult",
    "FileContentsResult",
]
