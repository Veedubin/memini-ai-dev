"""Audit logger for tracking security-relevant events.

Provides fire-and-forget asynchronous logging with batch INSERT to PostgreSQL.
Flushes at configurable size threshold (default 100) or interval (default 5s).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import asyncpg

from memini_ai.utils.logger import logger

# Valid event types (must match SQL CHECK constraint)
VALID_EVENT_TYPES = frozenset(
    {
        "auth_failure",
        "permission_change",
        "config_modification",
        "agent_execution",
        "memory_mutation",
        "tool_invocation",
        "trust_adjustment",
    }
)

# Valid severity levels (must match SQL CHECK constraint)
VALID_SEVERITIES = frozenset({"info", "warning", "critical"})


class EventType(StrEnum):
    """Audit event types matching SQL CHECK constraint."""

    AUTH_FAILURE = "auth_failure"
    PERMISSION_CHANGE = "permission_change"
    CONFIG_MODIFICATION = "config_modification"
    AGENT_EXECUTION = "agent_execution"
    MEMORY_MUTATION = "memory_mutation"
    TOOL_INVOCATION = "tool_invocation"
    TRUST_ADJUSTMENT = "trust_adjustment"


class Severity(StrEnum):
    """Audit severity levels matching SQL CHECK constraint."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit event before persistence."""

    event_type: str
    severity: str = "info"
    session_id: str | None = None
    peer_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    memory_id: str | None = None
    description: str | None = None
    details: dict[str, Any] | None = None
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    ip_address: str | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AuditLogger:
    """Asynchronous audit logger with batch INSERT to PostgreSQL.

    Fire-and-forget design: log() adds events to an internal buffer and
    returns immediately. A background task flushes events to the database
    in batches when either:
    - The buffer reaches flush_size events, OR
    - flush_interval seconds have elapsed since the last flush.

    Never raises exceptions from log() — all DB errors are caught and logged.
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool | None = None,
        flush_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize AuditLogger.

        Args:
            db_pool: Optional asyncpg connection pool. If None, logger operates
                in degraded mode (buffers events but never flushes).
            flush_size: Number of events to buffer before auto-flush.
            flush_interval: Seconds between periodic flushes.
        """
        self._db_pool = db_pool
        self._buffer: asyncio.Queue[AuditEvent] = asyncio.Queue()
        self._flush_size = flush_size
        self._flush_interval = flush_interval
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            "audit_logger_started",
            flush_size=self._flush_size,
            flush_interval=self._flush_interval,
        )

    async def stop(self) -> None:
        """Stop the background flush task and flush remaining events."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Final flush of any remaining events
        await self._flush()
        logger.info("audit_logger_stopped")

    def log(
        self,
        event_type: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> None:
        """Log an audit event (fire-and-forget).

        Adds the event to the internal buffer. Never raises exceptions.
        Invalid event_type or severity values are silently corrected to defaults.

        Args:
            event_type: Type of event (must be one of VALID_EVENT_TYPES).
            severity: Severity level (info, warning, critical).
            **kwargs: Additional AuditEvent fields (session_id, peer_id,
                agent_name, tool_name, memory_id, description, details,
                state_before, state_after, ip_address).
        """
        try:
            # Validate event_type
            if event_type not in VALID_EVENT_TYPES:
                logger.warning(
                    "audit_invalid_event_type",
                    event_type=event_type,
                    valid_types=list(VALID_EVENT_TYPES),
                )
                return

            # Validate severity
            if severity not in VALID_SEVERITIES:
                logger.warning(
                    "audit_invalid_severity",
                    severity=severity,
                    valid_severities=list(VALID_SEVERITIES),
                )
                severity = "info"

            event = AuditEvent(
                event_type=event_type,
                severity=severity,
                session_id=kwargs.get("session_id"),
                peer_id=kwargs.get("peer_id"),
                agent_name=kwargs.get("agent_name"),
                tool_name=kwargs.get("tool_name"),
                memory_id=kwargs.get("memory_id"),
                description=kwargs.get("description"),
                details=kwargs.get("details"),
                state_before=kwargs.get("state_before"),
                state_after=kwargs.get("state_after"),
                ip_address=kwargs.get("ip_address"),
            )
            self._buffer.put_nowait(event)

            # Trigger immediate flush if buffer is large enough
            if self._buffer.qsize() >= self._flush_size:
                asyncio.create_task(self._flush())
        except Exception:
            # Never let audit logging break the main flow
            logger.exception("audit_log_error")

    async def _flush_loop(self) -> None:
        """Background task that periodically flushes the buffer."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("audit_flush_loop_error")
                # Back off briefly on error
                await asyncio.sleep(1.0)

    async def _flush(self) -> None:
        """Flush all buffered events to the database via batch INSERT."""
        if self._db_pool is None:
            # Drain buffer silently in degraded mode
            while not self._buffer.empty():
                try:
                    self._buffer.get_nowait()
                except asyncio.QueueEmpty:
                    break
            return

        events: list[AuditEvent] = []
        while not self._buffer.empty():
            try:
                events.append(self._buffer.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not events:
            return

        try:
            async with self._db_pool.acquire() as conn:
                rows = []
                for event in events:
                    occurred_at = event.occurred_at
                    if occurred_at.tzinfo is None:
                        occurred_at = occurred_at.replace(tzinfo=UTC)

                    session_id = (
                        _to_uuid(event.session_id) if event.session_id else None
                    )
                    memory_id = _to_uuid(event.memory_id) if event.memory_id else None

                    rows.append(
                        (
                            event.event_type,
                            event.severity,
                            session_id,
                            event.peer_id,
                            event.agent_name,
                            event.tool_name,
                            memory_id,
                            event.description,
                            json.dumps(event.details) if event.details else None,
                            json.dumps(event.state_before)
                            if event.state_before
                            else None,
                            json.dumps(event.state_after)
                            if event.state_after
                            else None,
                            event.ip_address,
                            occurred_at,
                        )
                    )

                await conn.executemany(
                    """
                    INSERT INTO audit_log (
                        event_type, severity, session_id, peer_id, agent_name,
                        tool_name, memory_id, description, details,
                        state_before, state_after, ip_address, occurred_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb,
                              $10::jsonb, $11::jsonb, $12, $13)
                    """,
                    rows,
                )

            logger.debug("audit_flushed", event_count=len(events))
        except Exception:
            # Re-queue events on failure so they aren't lost
            for event in reversed(events):
                with contextlib.suppress(asyncio.QueueFull):
                    self._buffer.put_nowait(event)
            logger.exception("audit_flush_error", event_count=len(events))

    async def get_events(
        self,
        filters: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit_log with filters.

        Args:
            filters: Dictionary of filters. Supported keys:
                - event_type: Filter by event type
                - severity: Filter by severity level
                - agent_name: Filter by agent name
                - session_id: Filter by session ID
                - start_time: Filter events after this datetime
                - end_time: Filter events before this datetime
            limit: Maximum number of results (default 100).

        Returns:
            List of audit event dictionaries.
        """
        if self._db_pool is None:
            return []

        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if "event_type" in filters:
            conditions.append(f"event_type = ${param_idx}")
            params.append(filters["event_type"])
            param_idx += 1

        if "severity" in filters:
            conditions.append(f"severity = ${param_idx}")
            params.append(filters["severity"])
            param_idx += 1

        if "agent_name" in filters:
            conditions.append(f"agent_name = ${param_idx}")
            params.append(filters["agent_name"])
            param_idx += 1

        if "session_id" in filters:
            conditions.append(f"session_id = ${param_idx}")
            params.append(_to_uuid(filters["session_id"]))
            param_idx += 1

        if "start_time" in filters:
            conditions.append(f"occurred_at >= ${param_idx}")
            params.append(filters["start_time"])
            param_idx += 1

        if "end_time" in filters:
            conditions.append(f"occurred_at <= ${param_idx}")
            params.append(filters["end_time"])
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)
        limit_param = f"${param_idx}"

        query = f"""
            SELECT id, event_type, severity, session_id, peer_id, agent_name,
                   tool_name, memory_id, description, details, state_before,
                   state_after, ip_address, created_at, occurred_at
            FROM audit_log
            {where_clause}
            ORDER BY occurred_at DESC
            LIMIT {limit_param}
        """

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results: list[dict[str, Any]] = []
            for row in rows:
                result = dict(row)
                # Convert UUID and datetime fields to strings
                for key in ("id", "session_id", "memory_id"):
                    if result.get(key) is not None:
                        result[key] = str(result[key])
                for key in ("created_at", "occurred_at"):
                    if result.get(key) is not None:
                        result[key] = result[key].isoformat()
                results.append(result)
            return results
        except Exception:
            logger.exception("audit_get_events_error")
            return []

    async def get_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get aggregated metrics for the last N hours.

        Args:
            hours: Number of hours to look back (default 24).

        Returns:
            Dictionary with total_events, critical_count, events_per_agent,
            events_per_type, severity_counts.
        """
        if self._db_pool is None:
            return {
                "total_events": 0,
                "critical_count": 0,
                "events_per_agent": {},
                "events_per_type": {},
                "severity_counts": {},
            }

        try:
            async with self._db_pool.acquire() as conn:
                # Total events
                total = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM audit_log
                    WHERE occurred_at >= NOW() - ($1 || ' hours')::interval
                    """,
                    str(hours),
                )

                # Critical count
                critical = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM audit_log
                    WHERE severity = 'critical'
                    AND occurred_at >= NOW() - ($1 || ' hours')::interval
                    """,
                    str(hours),
                )

                # Events per agent
                agent_rows = await conn.fetch(
                    """
                    SELECT agent_name, COUNT(*) as count
                    FROM audit_log
                    WHERE occurred_at >= NOW() - ($1 || ' hours')::interval
                    AND agent_name IS NOT NULL
                    GROUP BY agent_name
                    ORDER BY count DESC
                    """,
                    str(hours),
                )
                events_per_agent = {
                    row["agent_name"]: row["count"] for row in agent_rows
                }

                # Events per type
                type_rows = await conn.fetch(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM audit_log
                    WHERE occurred_at >= NOW() - ($1 || ' hours')::interval
                    GROUP BY event_type
                    ORDER BY count DESC
                    """,
                    str(hours),
                )
                events_per_type = {row["event_type"]: row["count"] for row in type_rows}

                # Severity counts
                sev_rows = await conn.fetch(
                    """
                    SELECT severity, COUNT(*) as count
                    FROM audit_log
                    WHERE occurred_at >= NOW() - ($1 || ' hours')::interval
                    GROUP BY severity
                    """,
                    str(hours),
                )
                severity_counts = {row["severity"]: row["count"] for row in sev_rows}

            return {
                "total_events": total or 0,
                "critical_count": critical or 0,
                "events_per_agent": events_per_agent,
                "events_per_type": events_per_type,
                "severity_counts": severity_counts,
            }
        except Exception:
            logger.exception("audit_get_summary_error")
            return {
                "total_events": 0,
                "critical_count": 0,
                "events_per_agent": {},
                "events_per_type": {},
                "severity_counts": {},
            }


def _to_uuid(value: str | None) -> uuid.UUID | None:
    """Convert a string to UUID, returning None for invalid values.

    Args:
        value: String UUID or None.

    Returns:
        UUID object or None.
    """
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
