"""Tests for the kanban cards feature.

Covers:
- Schema constants (no vector column, PRIMARY KEY, UNIQUE, indexes, get_schema_sql)
- MemorySourceType enum (github, image)
- Database methods (integration-style against live DB)
- MCP tool layer (mocked)
- get_status kanbanCardCount field
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from memini_ai.memory.schema import MemorySourceType
from memini_ai.postgres.database import KANBAN_VALID_STATUSES, PostgresDatabase
from memini_ai.server import MCPServer

# =============================================================================
# Test card IDs — all test rows use this prefix for easy cleanup
# =============================================================================

TEST_CARD_PREFIX = "TEST-KB-"


def _test_card_id(suffix: str = "001") -> str:
    return f"{TEST_CARD_PREFIX}{suffix}"


# =============================================================================
# Group 1: Schema constants
# =============================================================================


class TestSchemaConstants:
    """Kanban table SQL contains no vector column; has PRIMARY KEY, UNIQUE,
    indexes; present in get_schema_sql() output."""

    def test_kanban_table_has_no_vector_column(self) -> None:
        """The kanban_cards table SQL must NOT contain 'vector'."""
        from memini_ai.postgres.schema import SQL_CREATE_KANBAN_CARDS_TABLE

        assert "vector" not in SQL_CREATE_KANBAN_CARDS_TABLE.lower()

    def test_kanban_table_has_primary_key(self) -> None:
        """The kanban_cards table must have a PRIMARY KEY on card_id."""
        from memini_ai.postgres.schema import SQL_CREATE_KANBAN_CARDS_TABLE

        assert "PRIMARY KEY" in SQL_CREATE_KANBAN_CARDS_TABLE
        assert "card_id" in SQL_CREATE_KANBAN_CARDS_TABLE

    def test_kanban_table_has_unique_constraint(self) -> None:
        """The kanban_cards table must have UNIQUE (repo, number, item_type)."""
        from memini_ai.postgres.schema import SQL_CREATE_KANBAN_CARDS_TABLE

        assert "UNIQUE" in SQL_CREATE_KANBAN_CARDS_TABLE
        assert "repo" in SQL_CREATE_KANBAN_CARDS_TABLE
        assert "number" in SQL_CREATE_KANBAN_CARDS_TABLE
        assert "item_type" in SQL_CREATE_KANBAN_CARDS_TABLE

    def test_kanban_indexes_exist(self) -> None:
        """Indexes on status, repo, and created_at must exist."""
        from memini_ai.postgres.schema import SQL_CREATE_KANBAN_CARDS_INDEXES

        assert "idx_kanban_cards_status" in SQL_CREATE_KANBAN_CARDS_INDEXES
        assert "idx_kanban_cards_repo" in SQL_CREATE_KANBAN_CARDS_INDEXES
        assert "idx_kanban_cards_created" in SQL_CREATE_KANBAN_CARDS_INDEXES

    def test_kanban_table_in_get_schema_sql(self) -> None:
        """get_schema_sql() must include the kanban_cards table SQL."""
        from memini_ai.postgres.schema import get_schema_sql

        sql = get_schema_sql()
        assert "kanban_cards" in sql
        assert "card_id" in sql
        assert "repo" in sql
        assert "number" in sql
        assert "item_type" in sql

    def test_kanban_table_name_constant(self) -> None:
        """TABLE_KANBAN_CARDS must be 'kanban_cards'."""
        from memini_ai.postgres.schema import TABLE_KANBAN_CARDS

        assert TABLE_KANBAN_CARDS == "kanban_cards"

    def test_kanban_valid_statuses(self) -> None:
        """KANBAN_VALID_STATUSES must match the DB CHECK constraint."""
        expected = {"triage", "todo", "ready", "running", "blocked", "done", "archived"}
        assert expected == KANBAN_VALID_STATUSES


# =============================================================================
# Group 2: Source type enum
# =============================================================================


class TestMemorySourceTypeKanban:
    """'github' and 'image' are valid MemorySourceType values."""

    def test_github_source_type(self) -> None:
        """MemorySourceType.github == 'github'."""
        assert MemorySourceType.github == "github"

    def test_image_source_type(self) -> None:
        """MemorySourceType.image == 'image'."""
        assert MemorySourceType.image == "image"

    def test_github_is_valid_enum_value(self) -> None:
        """MemorySourceType('github') works."""
        st = MemorySourceType("github")
        assert st == MemorySourceType.github

    def test_image_is_valid_enum_value(self) -> None:
        """MemorySourceType('image') works."""
        st = MemorySourceType("image")
        assert st == MemorySourceType.image

    def test_all_source_types_includes_github_and_image(self) -> None:
        """All source types include github and image."""
        values = {e.value for e in MemorySourceType}
        assert "github" in values
        assert "image" in values


# =============================================================================
# Group 3: Database methods (integration-style against live DB)
# =============================================================================


class TestDatabaseKanban:
    """Integration tests for kanban database methods.

    Uses the pg_db fixture (connects to live memini-postgres on port 5434).
    All test rows are cleaned up after each test.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def _cleanup_kanban(self, pg_db: PostgresDatabase) -> None:
        """Pre-test and post-test cleanup of test kanban rows."""
        # Pre-test cleanup
        try:
            async with pg_db._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM kanban_cards WHERE card_id LIKE '{TEST_CARD_PREFIX}%'"
                )
        except Exception:
            pass
        yield
        # Post-test cleanup
        try:
            async with pg_db._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM kanban_cards WHERE card_id LIKE '{TEST_CARD_PREFIX}%'"
                )
        except Exception:
            pass

    async def test_add_kanban_card_returns_dict(self, pg_db: PostgresDatabase) -> None:
        """add_kanban_card returns a dict with card_id and inserted=True."""
        card_id = _test_card_id("add-001")
        result = await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo",
            number=1,
            item_type="bug",
            url="https://github.com/test/test-repo/issues/1",
            title="Test bug",
            author="tester",
        )
        assert isinstance(result, dict)
        assert result["card_id"] == card_id
        assert result["inserted"] is True
        assert result["repo"] == "test-repo"
        assert result["number"] == 1
        assert result["item_type"] == "bug"
        assert result["status"] == "triage"
        assert result["title"] == "Test bug"
        assert result["author"] == "tester"

    async def test_add_kanban_card_duplicate_idempotent(
        self, pg_db: PostgresDatabase
    ) -> None:
        """Duplicate (repo, number, item_type) is idempotent — no error, no dup row."""
        card_id = _test_card_id("dup-001")
        # First insert
        result1 = await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-dup",
            number=42,
            item_type="feature",
            url="https://github.com/test/test-repo-dup/issues/42",
            title="First insert",
        )
        assert result1["inserted"] is True

        # Second insert — same (repo, number, item_type) — should be idempotent
        result2 = await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-dup",
            number=42,
            item_type="feature",
            url="https://github.com/test/test-repo-dup/issues/42",
            title="First insert",
        )
        assert result2["inserted"] is False

        # Verify the card is fetchable and there's no duplicate error.
        fetched = await pg_db.get_kanban_card(card_id)
        assert fetched is not None
        assert fetched["card_id"] == card_id

    async def test_move_kanban_card_updates_status(
        self, pg_db: PostgresDatabase
    ) -> None:
        """move_kanban_card updates status and updated_at."""
        card_id = _test_card_id("move-001")
        await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-move",
            number=10,
            item_type="pr",
            url="https://github.com/test/test-repo-move/pull/10",
            title="Test PR",
        )

        # Move from triage -> ready
        moved = await pg_db.move_kanban_card(card_id, "ready")
        assert moved is not None
        assert moved["card_id"] == card_id
        assert moved["status"] == "ready"
        assert moved["updated_at"] is not None

        # Verify via get
        fetched = await pg_db.get_kanban_card(card_id)
        assert fetched is not None
        assert fetched["status"] == "ready"

    async def test_move_kanban_card_invalid_status_raises(
        self, pg_db: PostgresDatabase
    ) -> None:
        """move_kanban_card with invalid status raises ValueError."""
        card_id = _test_card_id("invalid-001")
        await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-invalid",
            number=99,
            item_type="bug",
            url="https://github.com/test/test-repo-invalid/issues/99",
            title="Invalid move test",
        )

        with pytest.raises(ValueError, match="Invalid kanban status"):
            await pg_db.move_kanban_card(card_id, "nonexistent")

    async def test_move_kanban_card_not_found_returns_none(
        self, pg_db: PostgresDatabase
    ) -> None:
        """move_kanban_card returns None for nonexistent card_id."""
        result = await pg_db.move_kanban_card("nonexistent-card-id", "done")
        assert result is None

    async def test_list_kanban_cards_filters_by_status(
        self, pg_db: PostgresDatabase
    ) -> None:
        """list_kanban_cards filters by status."""
        # Insert two cards with different statuses
        await pg_db.add_kanban_card(
            card_id=_test_card_id("list-status-001"),
            repo="test-repo-list",
            number=1,
            item_type="bug",
            url="https://github.com/test/test-repo-list/issues/1",
            title="Bug in triage",
        )
        await pg_db.add_kanban_card(
            card_id=_test_card_id("list-status-002"),
            repo="test-repo-list",
            number=2,
            item_type="feature",
            url="https://github.com/test/test-repo-list/issues/2",
            title="Feature in todo",
        )
        # Move the second to 'todo'
        await pg_db.move_kanban_card(_test_card_id("list-status-002"), "todo")

        # Filter by status='triage'
        triage_cards = await pg_db.list_kanban_cards(status="triage")
        triage_ids = {c["card_id"] for c in triage_cards}
        assert _test_card_id("list-status-001") in triage_ids
        assert _test_card_id("list-status-002") not in triage_ids

        # Filter by status='todo'
        todo_cards = await pg_db.list_kanban_cards(status="todo")
        todo_ids = {c["card_id"] for c in todo_cards}
        assert _test_card_id("list-status-002") in todo_ids
        assert _test_card_id("list-status-001") not in todo_ids

    async def test_list_kanban_cards_filters_by_repo(
        self, pg_db: PostgresDatabase
    ) -> None:
        """list_kanban_cards filters by repo."""
        await pg_db.add_kanban_card(
            card_id=_test_card_id("list-repo-001"),
            repo="repo-alpha",
            number=1,
            item_type="bug",
            url="https://github.com/test/repo-alpha/issues/1",
            title="Alpha bug",
        )
        await pg_db.add_kanban_card(
            card_id=_test_card_id("list-repo-002"),
            repo="repo-beta",
            number=1,
            item_type="bug",
            url="https://github.com/test/repo-beta/issues/1",
            title="Beta bug",
        )

        alpha_cards = await pg_db.list_kanban_cards(repo="repo-alpha")
        alpha_ids = {c["card_id"] for c in alpha_cards}
        assert _test_card_id("list-repo-001") in alpha_ids
        assert _test_card_id("list-repo-002") not in alpha_ids

    async def test_get_kanban_card_returns_card(self, pg_db: PostgresDatabase) -> None:
        """get_kanban_card returns the card dict for an existing card."""
        card_id = _test_card_id("get-001")
        await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-get",
            number=7,
            item_type="question",
            url="https://github.com/test/test-repo-get/issues/7",
            title="A question",
            author="asker",
            wrapped_text="Wrapped prompt text here",
        )

        card = await pg_db.get_kanban_card(card_id)
        assert card is not None
        assert card["card_id"] == card_id
        assert card["title"] == "A question"
        assert card["author"] == "asker"
        assert card["wrapped_text"] == "Wrapped prompt text here"
        assert card["item_type"] == "question"

    async def test_get_kanban_card_not_found_returns_none(
        self, pg_db: PostgresDatabase
    ) -> None:
        """get_kanban_card returns None for nonexistent card_id."""
        card = await pg_db.get_kanban_card("nonexistent-card-id")
        assert card is None

    async def test_count_kanban_cards_increments(self, pg_db: PostgresDatabase) -> None:
        """count_kanban_cards increments as cards are added."""
        before = await pg_db.count_kanban_cards()

        await pg_db.add_kanban_card(
            card_id=_test_card_id("count-001"),
            repo="test-repo-count",
            number=1,
            item_type="docs",
            url="https://github.com/test/test-repo-count/issues/1",
            title="Doc card",
        )
        after_one = await pg_db.count_kanban_cards()
        assert after_one == before + 1

        await pg_db.add_kanban_card(
            card_id=_test_card_id("count-002"),
            repo="test-repo-count",
            number=2,
            item_type="bug",
            url="https://github.com/test/test-repo-count/issues/2",
            title="Bug card",
        )
        after_two = await pg_db.count_kanban_cards()
        assert after_two == before + 2

    async def test_add_kanban_card_with_draft_flag(
        self, pg_db: PostgresDatabase
    ) -> None:
        """add_kanban_card stores the draft flag correctly."""
        card_id = _test_card_id("draft-001")
        result = await pg_db.add_kanban_card(
            card_id=card_id,
            repo="test-repo-draft",
            number=1,
            item_type="pr",
            url="https://github.com/test/test-repo-draft/pull/1",
            title="Draft PR",
            draft=True,
        )
        assert result["draft"] is True

        # Non-draft
        card_id2 = _test_card_id("draft-002")
        result2 = await pg_db.add_kanban_card(
            card_id=card_id2,
            repo="test-repo-draft",
            number=2,
            item_type="pr",
            url="https://github.com/test/test-repo-draft/pull/2",
            title="Non-draft PR",
            draft=False,
        )
        assert result2["draft"] is False


# =============================================================================
# Group 4: MCP tool layer
# =============================================================================


class _MockDb:
    """A mock database that supports kanban operations."""

    def __init__(self) -> None:
        self.cards: dict[str, dict] = {}
        self.add_kanban_card = AsyncMock()
        self.move_kanban_card = AsyncMock()
        self.list_kanban_cards = AsyncMock(return_value=[])
        self.get_kanban_card = AsyncMock()
        self.count_kanban_cards = AsyncMock(return_value=0)
        self.count_memories = AsyncMock(return_value=42)
        self.count_thoughts = AsyncMock(return_value=7)


@pytest.fixture
def mcp_server() -> MCPServer:
    """Create an MCPServer instance for testing."""
    return MCPServer()


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem with kanban-capable _db."""
    mock = MagicMock()
    mock.is_ready = True
    mock.is_initialized = True
    mock._db = _MockDb()
    mock.query_memories = AsyncMock(return_value=[])
    mock.add_memory = AsyncMock(return_value="test-memory-id-123")
    mock.get_memory = AsyncMock(return_value=MagicMock(id="test-memory-id-123"))
    return mock


class TestKanbanAddCardTool:
    """Tests for the kanban_add_card MCP tool."""

    @pytest.mark.asyncio
    async def test_add_card_success(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_add_card returns success with card data."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.add_kanban_card = AsyncMock(
            return_value={
                "card_id": "T-GH-001",
                "repo": "test-repo",
                "number": 1,
                "item_type": "bug",
                "status": "triage",
                "url": "https://github.com/test/test-repo/issues/1",
                "title": "Test bug",
                "author": "tester",
                "wrapped_text": None,
                "draft": False,
                "memory_id": None,
                "created_at": None,
                "updated_at": None,
                "inserted": True,
            }
        )

        result = await mcp_server.kanban_add_card(
            card_id="T-GH-001",
            repo="test-repo",
            number=1,
            item_type="bug",
            url="https://github.com/test/test-repo/issues/1",
            title="Test bug",
        )

        assert result["success"] is True
        assert result["card_id"] == "T-GH-001"
        assert result["inserted"] is True

    @pytest.mark.asyncio
    async def test_add_card_db_error(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_add_card returns error shape on DB failure."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.add_kanban_card = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        result = await mcp_server.kanban_add_card(
            card_id="T-GH-ERR",
            repo="test-repo",
            number=999,
            item_type="bug",
            url="https://github.com/test/test-repo/issues/999",
            title="Error test",
        )

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
        assert "connection lost" in result["error"]


class TestKanbanMoveCardTool:
    """Tests for the kanban_move_card MCP tool."""

    @pytest.mark.asyncio
    async def test_move_card_success(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_move_card returns success with updated card."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.move_kanban_card = AsyncMock(
            return_value={
                "card_id": "T-GH-001",
                "status": "done",
                "repo": "test-repo",
                "number": 1,
                "item_type": "bug",
                "url": "https://github.com/test/test-repo/issues/1",
                "title": "Test bug",
                "author": "tester",
                "wrapped_text": None,
                "draft": False,
                "memory_id": None,
                "created_at": None,
                "updated_at": None,
            }
        )

        result = await mcp_server.kanban_move_card(card_id="T-GH-001", status="done")

        assert result["success"] is True
        assert result["status"] == "done"

    @pytest.mark.asyncio
    async def test_move_card_invalid_status(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_move_card with invalid status returns error shape."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.move_kanban_card = AsyncMock(
            side_effect=ValueError(
                "Invalid kanban status 'bogus'. "
                "Must be one of: ['archived', 'blocked', 'done', 'ready', "
                "'running', 'todo', 'triage']"
            )
        )

        result = await mcp_server.kanban_move_card(card_id="T-GH-001", status="bogus")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
        assert "Invalid kanban status" in result["error"]

    @pytest.mark.asyncio
    async def test_move_card_not_found(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_move_card for nonexistent card returns error shape."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.move_kanban_card = AsyncMock(return_value=None)

        result = await mcp_server.kanban_move_card(card_id="nonexistent", status="done")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
        assert "not found" in result["error"].lower()


class TestKanbanListCardsTool:
    """Tests for the kanban_list_cards MCP tool."""

    @pytest.mark.asyncio
    async def test_list_cards_success(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_list_cards returns cards list and count."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.list_kanban_cards = AsyncMock(
            return_value=[
                {
                    "card_id": "T-GH-001",
                    "repo": "test-repo",
                    "number": 1,
                    "item_type": "bug",
                    "status": "triage",
                    "url": "https://github.com/test/test-repo/issues/1",
                    "title": "Bug 1",
                    "author": "tester",
                    "wrapped_text": None,
                    "draft": False,
                    "memory_id": None,
                    "created_at": None,
                    "updated_at": None,
                },
            ]
        )

        result = await mcp_server.kanban_list_cards()

        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["cards"]) == 1
        assert result["cards"][0]["card_id"] == "T-GH-001"

    @pytest.mark.asyncio
    async def test_list_cards_with_filters(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_list_cards passes status and repo filters to DB."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.list_kanban_cards = AsyncMock(return_value=[])

        result = await mcp_server.kanban_list_cards(
            status="todo", repo="my-repo", limit=50
        )

        assert result["success"] is True
        assert result["count"] == 0
        # Verify the DB method was called with the right filters
        mock_memory_system._db.list_kanban_cards.assert_called_once_with(
            status="todo", repo="my-repo", limit=50
        )


class TestKanbanGetCardTool:
    """Tests for the kanban_get_card MCP tool."""

    @pytest.mark.asyncio
    async def test_get_card_success(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_get_card returns card data for existing card."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.get_kanban_card = AsyncMock(
            return_value={
                "card_id": "T-GH-001",
                "repo": "test-repo",
                "number": 1,
                "item_type": "bug",
                "status": "triage",
                "url": "https://github.com/test/test-repo/issues/1",
                "title": "Test bug",
                "author": "tester",
                "wrapped_text": None,
                "draft": False,
                "memory_id": None,
                "created_at": None,
                "updated_at": None,
            }
        )

        result = await mcp_server.kanban_get_card(card_id="T-GH-001")

        assert result["success"] is True
        assert result["card_id"] == "T-GH-001"

    @pytest.mark.asyncio
    async def test_get_card_not_found(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """kanban_get_card for nonexistent card returns error shape."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.get_kanban_card = AsyncMock(return_value=None)

        result = await mcp_server.kanban_get_card(card_id="nonexistent")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
        assert "not found" in result["error"].lower()


# =============================================================================
# Group 5: get_status includes kanbanCardCount
# =============================================================================


class TestGetStatusKanbanCount:
    """get_status includes kanbanCardCount key (int)."""

    @pytest.mark.asyncio
    async def test_get_status_includes_kanban_card_count(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """get_status must include kanbanCardCount as an int."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.count_kanban_cards = AsyncMock(return_value=5)

        result = await mcp_server.get_status()

        assert "kanbanCardCount" in result
        assert isinstance(result["kanbanCardCount"], int)
        assert result["kanbanCardCount"] == 5

    @pytest.mark.asyncio
    async def test_get_status_kanban_count_defaults_to_zero(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """When count_kanban_cards raises, kanbanCardCount must be 0."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system._db.count_kanban_cards = AsyncMock(
            side_effect=RuntimeError("count failed")
        )

        result = await mcp_server.get_status()

        assert "kanbanCardCount" in result
        assert result["kanbanCardCount"] == 0

    @pytest.mark.asyncio
    async def test_get_status_kanban_count_when_db_missing_method(
        self,
        mcp_server: MCPServer,
    ) -> None:
        """When _db has no count_kanban_cards, kanbanCardCount must be 0."""
        mock = MagicMock()
        mock.is_ready = True
        mock.is_initialized = True
        mock._db = MagicMock()  # No count_kanban_cards method
        mock.query_memories = AsyncMock(return_value=[])
        mock.add_memory = AsyncMock(return_value="test-id")
        mock.get_memory = AsyncMock(return_value=MagicMock(id="test-id"))
        mcp_server._memory_system = mock

        result = await mcp_server.get_status()

        assert "kanbanCardCount" in result
        assert result["kanbanCardCount"] == 0
