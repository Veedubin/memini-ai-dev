"""Rate limiter for memory write operations.

Implements a sliding window rate limiter to prevent resource exhaustion
from excessive add_memory calls. Uses per-peer tracking with configurable limits.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from memini_ai.utils.logger import logger
from memini_ai.utils.sanitizer import RateLimitExceededError


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for add_memory operations.

    Tracks request counts per peer ID using a sliding time window.
    When a peer exceeds the configured limit within the window,
    subsequent requests are rejected until the window slides past
    older requests.

    This implementation uses an asyncio.Lock for thread safety in
    async contexts and stores timestamps in a per-peer deque.

    Args:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Size of the sliding window in seconds (default 60).
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def _cleanup_window(self, peer_id: str, now: float) -> None:
        """Remove timestamps outside the current window for a peer.

        Args:
            peer_id: The peer identifier.
            now: Current time in seconds since epoch.
        """
        cutoff = now - self.window_seconds
        timestamps = self._timestamps.get(peer_id, [])
        # Filter to only timestamps within the window
        self._timestamps[peer_id] = [ts for ts in timestamps if ts > cutoff]

    def check_rate_limit(self, peer_id: str) -> tuple[bool, int, int]:
        """Check if a request is allowed under the rate limit.

        Uses a sliding window approach: removes old timestamps outside
        the window, then checks if adding a new request would exceed
        the limit.

        Args:
            peer_id: Identifier for the client/peer making the request.

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds):
            - allowed: True if the request is permitted
            - remaining: Number of requests remaining in this window
            - retry_after_seconds: Seconds until the next slot opens (0 if allowed)
        """
        now = time.monotonic()
        self._cleanup_window(peer_id, now)

        current_count = len(self._timestamps.get(peer_id, []))

        if current_count >= self.max_requests:
            # Calculate when the oldest request in the window will expire
            oldest = self._timestamps[peer_id][0] if self._timestamps[peer_id] else now
            retry_after = max(0.0, oldest + self.window_seconds - now)
            remaining = 0
            return False, remaining, int(retry_after) + 1

        # Record the new request
        self._timestamps[peer_id].append(now)
        remaining = self.max_requests - current_count - 1
        return True, remaining, 0

    def check_and_raise(self, peer_id: str) -> None:
        """Check rate limit and raise if exceeded.

        Convenience method that raises RateLimitExceededError
        if the request would exceed the limit.

        Args:
            peer_id: Identifier for the client/peer making the request.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """
        allowed, remaining, retry_after = self.check_rate_limit(peer_id)
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                peer_id=peer_id,
                max_requests=self.max_requests,
                window_seconds=self.window_seconds,
                retry_after=retry_after,
            )
            raise RateLimitExceededError(
                peer_id=peer_id,
                limit=self.max_requests,
                window_seconds=self.window_seconds,
            )

    def reset(self, peer_id: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            peer_id: If provided, reset only this peer. If None, reset all.
        """
        if peer_id is not None:
            self._timestamps.pop(peer_id, None)
        else:
            self._timestamps.clear()


class AsyncRateLimiter:
    """Async wrapper around SlidingWindowRateLimiter for use with FastMCP.

    Provides an async-safe interface for rate limiting in async contexts.
    Uses asyncio.Lock to ensure consistency across concurrent requests.

    Args:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self._limiter = SlidingWindowRateLimiter(max_requests, window_seconds)
        self._lock: Any = None  # asyncio.Lock, created lazily

    async def _ensure_lock(self) -> Any:
        """Lazily create asyncio.Lock in the running event loop."""
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check_rate_limit(self, peer_id: str) -> tuple[bool, int, int]:
        """Async rate limit check.

        Args:
            peer_id: Identifier for the client/peer making the request.

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds).
        """
        lock = await self._ensure_lock()
        async with lock:
            return self._limiter.check_rate_limit(peer_id)

    async def check_and_raise(self, peer_id: str) -> None:
        """Async rate limit check that raises on exceeded.

        Args:
            peer_id: Identifier for the client/peer making the request.

        Raises:
            RateLimitExceededError: If the rate limit is exceeded.
        """
        lock = await self._ensure_lock()
        async with lock:
            self._limiter.check_and_raise(peer_id)

    async def reset(self, peer_id: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            peer_id: If provided, reset only this peer. If None, reset all.
        """
        lock = await self._ensure_lock()
        async with lock:
            self._limiter.reset(peer_id)
