"""File watcher using watchdog for event-based file monitoring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from watchdog.events import (
    DirDeletedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)

if TYPE_CHECKING:
    pass

from memini_ai.indexer.constants import ALLOWED_EXTENSIONS, ALWAYS_EXCLUDED, SKIP_DIRS


@runtime_checkable
class FileChangeCallback(Protocol):
    """Callback protocol for file change events."""

    async def on_file_created(self, path: str) -> None: ...
    async def on_file_modified(self, path: str) -> None: ...
    async def on_file_deleted(self, path: str) -> None: ...
    async def on_file_moved(self, old_path: str, new_path: str) -> None: ...


@dataclass
class FileWatcher:
    """Event-based file monitoring using watchdog.

    Watches a directory tree for file changes and invokes callbacks
    for created, modified, deleted, and moved events.
    """

    _root_path: str
    _callback: FileChangeCallback
    _observer: Any = field(default=None, init=False)
    _debounce_delay: float = 0.1  # 100ms debounce
    _pending_events: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _running: bool = field(default=False)

    def __init__(
        self, root_path: str, callback: FileChangeCallback, debounce_ms: int = 100
    ) -> None:
        """Initialize file watcher.

        Args:
            root_path: Root directory to watch.
            callback: Callback for file change events.
            debounce_ms: Debounce delay in milliseconds.
        """
        self._root_path = str(Path(root_path).resolve())
        self._callback = callback
        self._debounce_delay = debounce_ms / 1000.0

    def _should_skip_path(self, path: str) -> bool:
        """Check if a path should be skipped.

        Args:
            path: Path to check.

        Returns:
            True if path should be skipped.
        """
        path_obj = Path(path)
        parts = path_obj.parts

        # Check if any part is in skip dirs
        for part in parts:
            if part in SKIP_DIRS:
                return True

        # Check for always excluded
        for pattern in ALWAYS_EXCLUDED:
            if pattern.startswith("~$"):
                # Special pattern for temp files
                if path_obj.name.startswith(pattern[1:]):
                    return True
            elif path_obj.name == pattern:
                return True

        return False

    def _is_allowed_file(self, path: str) -> bool:
        """Check if a file has an allowed extension.

        Args:
            path: Path to check.

        Returns:
            True if file is allowed.
        """
        ext = Path(path).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    async def start(self) -> None:
        """Start the file watcher."""
        if self._running:
            return

        from watchdog.observers import Observer

        self._running = True
        observer = Observer()
        handler = _FileWatcherHandler(self)
        observer.schedule(handler, self._root_path, recursive=True)
        observer.start()
        self._observer = observer

    async def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False

        # Cancel pending events
        for task in self._pending_events.values():
            task.cancel()
        self._pending_events.clear()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    async def _handle_event(self, event: FileSystemEvent) -> None:
        """Handle a file system event with debouncing.

        Args:
            event: The file system event.
        """
        if isinstance(event, DirDeletedEvent):
            return

        path = event.src_path
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        path_str = str(Path(path).resolve())

        # Skip non-files and excluded paths
        if not path_str or self._should_skip_path(path_str):
            return

        if event.event_type in (
            "created",
            "modified",
            "deleted",
            "moved",
        ) and not self._is_allowed_file(path_str):
            return

        # Debounce: cancel previous event for same path
        async with self._lock:
            existing = self._pending_events.get(path_str)
            if existing:
                existing.cancel()

            # Schedule new event
            task = asyncio.create_task(self._debounced_handle(event))
            self._pending_events[path_str] = task
            task.add_done_callback(lambda t: self._pending_events.pop(path_str, None))

    async def _debounced_handle(self, event: FileSystemEvent) -> None:
        """Handle event after debounce delay.

        Args:
            event: The file system event.
        """
        await asyncio.sleep(self._debounce_delay)

        if not self._running:
            return

        try:
            src_path = event.src_path
            if isinstance(src_path, bytes):
                src_path = src_path.decode("utf-8", errors="replace")
            src_path = str(Path(src_path).resolve())

            if isinstance(event, FileCreatedEvent):
                await self._callback.on_file_created(src_path)
            elif isinstance(event, FileModifiedEvent):
                await self._callback.on_file_modified(src_path)
            elif isinstance(event, FileDeletedEvent):
                await self._callback.on_file_deleted(src_path)
            elif isinstance(event, FileMovedEvent):
                dest_path = event.dest_path
                if isinstance(dest_path, bytes):
                    dest_path = dest_path.decode("utf-8", errors="replace")
                dest_path = str(Path(dest_path).resolve())
                await self._callback.on_file_moved(src_path, dest_path)
        except Exception:
            pass  # Log in production

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running and self._observer is not None


class _FileWatcherHandler(FileSystemEventHandler):
    """Internal handler for watchdog events."""

    def __init__(self, watcher: FileWatcher) -> None:
        self._watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event."""
        asyncio.create_task(self._watcher._handle_event(event))
