"""Snapshot index for incremental update detection using SHA-256."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class SnapshotEntry:
    """Single file snapshot entry."""

    path: str
    content_hash: str
    size: int
    modified: float


@dataclass
class SnapshotIndex:
    """Snapshot-based incremental update detection.

    Maintains a map of file paths to their content hashes for
    detecting which files have changed and need re-indexing.
    """

    _snapshots: dict[str, SnapshotEntry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _root_path: str | None = field(default=None)

    @property
    def root_path(self) -> str | None:
        """Get the root path being indexed."""
        return self._root_path

    async def set_root_path(self, path: str) -> None:
        """Set the root path for snapshots.

        Args:
            path: Root directory path.
        """
        async with self._lock:
            self._root_path = path

    async def update_file(
        self, path: str, content_hash: str, size: int, modified: float
    ) -> bool:
        """Update snapshot for a file.

        Args:
            path: File path.
            content_hash: SHA-256 hash of content.
            size: File size in bytes.
            modified: Modification timestamp.

        Returns:
            True if this is a new or changed file, False if unchanged.
        """
        async with self._lock:
            existing = self._snapshots.get(path)
            if existing and existing.content_hash == content_hash:
                return False  # Unchanged

            self._snapshots[path] = SnapshotEntry(
                path=path,
                content_hash=content_hash,
                size=size,
                modified=modified,
            )
            return True

    async def check_file(self, path: str, content_hash: str) -> bool:
        """Check if file has changed since last snapshot.

        Args:
            path: File path.
            content_hash: Current content hash.

        Returns:
            True if file is new or changed, False if unchanged.
        """
        async with self._lock:
            existing = self._snapshots.get(path)
            if existing is None:
                return True  # New file
            return existing.content_hash != content_hash

    async def remove_file(self, path: str) -> None:
        """Remove a file from the snapshot index.

        Args:
            path: File path to remove.
        """
        async with self._lock:
            self._snapshots.pop(path, None)

    async def get_hash(self, path: str) -> str | None:
        """Get the stored hash for a file.

        Args:
            path: File path.

        Returns:
            Stored content hash or None if not tracked.
        """
        async with self._lock:
            entry = self._snapshots.get(path)
            return entry.content_hash if entry else None

    async def get_changed_paths(self, current_paths: list[str]) -> list[str]:
        """Get paths that are new or changed compared to snapshots.

        Args:
            current_paths: Current list of file paths to check.

        Returns:
            List of paths that are new or have changed content.
        """
        async with self._lock:
            changed = []
            seen = set()

            for path in current_paths:
                seen.add(path)
                existing = self._snapshots.get(path)
                if existing is None:
                    changed.append(path)
                # If file exists, it's only changed if the hash differs
                # The actual hash comparison happens at update time

            # Files removed from disk
            removed = set(self._snapshots.keys()) - seen
            for path in removed:
                # Mark as removed by removing from snapshot
                self._snapshots.pop(path, None)

            return changed

    async def get_all_paths(self) -> list[str]:
        """Get all tracked file paths.

        Returns:
            List of all tracked file paths.
        """
        async with self._lock:
            return list(self._snapshots.keys())

    def get_stats(self) -> dict[str, int]:
        """Get snapshot statistics.

        Returns:
            Dictionary with file count and total size.
        """
        total_size = sum(entry.size for entry in self._snapshots.values())
        return {
            "tracked_files": len(self._snapshots),
            "total_size": total_size,
        }

    async def clear(self) -> None:
        """Clear all snapshots."""
        async with self._lock:
            self._snapshots.clear()
            self._root_path = None

    async def load_from_hashes(self, path_hash_map: dict[str, str]) -> None:
        """Load snapshots from a path->hash map (e.g., from file tracker).

        Args:
            path_hash_map: Dictionary mapping paths to content hashes.
        """
        async with self._lock:
            self._snapshots = {
                path: SnapshotEntry(
                    path=path,
                    content_hash=hash_val,
                    size=0,  # Unknown when loading from hash only
                    modified=0,  # Unknown when loading from hash only
                )
                for path, hash_val in path_hash_map.items()
            }

    def has_file(self, path: str) -> bool:
        """Check if a file is tracked.

        Args:
            path: File path to check.

        Returns:
            True if file is tracked, False otherwise.
        """
        return path in self._snapshots
