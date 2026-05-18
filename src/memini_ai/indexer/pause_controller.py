"""Pause controller for coordinating indexer pause/resume operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum


class PauseState(Enum):
    """State of the pause controller."""

    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"


@dataclass
class PauseController:
    """Async pause controller for coordinating indexer operations.

    Allows pausing indexing operations during priority tasks like
    search or memory operations. Uses an async condition to signal
    when pause state changes.
    """

    _state: PauseState = field(default=PauseState.RUNNING)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    _pause_count: int = field(default=0)
    _running_tasks: set[asyncio.Task[None]] = field(default_factory=set)

    @property
    def state(self) -> PauseState:
        """Current pause state."""
        return self._state

    @property
    def is_paused(self) -> bool:
        """Check if indexer is currently paused."""
        return self._state == PauseState.PAUSED

    async def pause(self) -> None:
        """Pause all indexing operations.

        Increments pause count and signals all running tasks to stop.
        """
        self._pause_count += 1
        self._state = PauseState.PAUSED
        async with self._cond:
            self._cond.notify_all()

    async def resume(self) -> None:
        """Resume indexing operations.

        Decrements pause count. Only resumes when pause_count reaches 0.
        """
        if self._pause_count > 0:
            self._pause_count -= 1
        if self._pause_count == 0:
            self._state = PauseState.RUNNING
            async with self._cond:
                self._cond.notify_all()

    async def wait_while_paused(self) -> None:
        """Wait until the indexer is not paused.

        This is a cooperative wait - tasks should await this
        periodically to check if they should stop.
        """
        if self._state == PauseState.PAUSED:
            async with self._cond:
                await self._cond.wait()

    def register_task(self, task: asyncio.Task[None]) -> None:
        """Register a task for tracking.

        Args:
            task: The task to register for tracking.
        """
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    async def cancel_all(self) -> None:
        """Cancel all running indexing tasks."""
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
