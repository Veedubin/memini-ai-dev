"""Tests for v1.6.0 tool-surface gating (MEMINI_TOOL_GROUPS).

Strategy: construct MCPServer with a patched get_config, then swap in a
recording FastMCP stand-in and re-run _setup_tools() to capture exactly
which tool functions would be registered per profile.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memini_ai.server import MCPServer

DEFAULT_EXPECTED: set[str] = {
    "query_memories",
    "add_memory",
    "search_project",
    "index_project",
    "get_file_contents",
    "get_status",
    "healthcheck",
    "get_trust_score",
    "adjust_trust",
    "preconpress_extraction",
    "get_tier0_summary",
    "get_tier1_summary",
    "kanban_add_card",
    "kanban_move_card",
    "kanban_list_cards",
    "kanban_get_card",
}

CHAIN_TRIO: set[str] = {"add_thought", "get_thought_chain", "get_related_chains"}

REMOVED_CHAIN_TOOLS: set[str] = {
    "start_thought_chain",
    "revise_thought",
    "branch_thought",
    "pause_thought_chain",
    "resume_thought_chain",
    "abandon_thought_chain",
}

CORE_ONLY: set[str] = {
    "query_memories",
    "add_memory",
    "search_project",
    "index_project",
    "get_file_contents",
    "get_status",
    "healthcheck",
}

DIALECTIC_TOOLS: set[str] = {
    "find_contradictions",
    "resolve_contradiction",
    "get_dialectic_history",
    "challenge_memory",
}

KG_TOOLS: set[str] = {
    "query_kg",
    "extract_entities",
    "get_entity_graph",
    "get_inference_chain",
    "search_entities",
    "get_graph_visualization",
}


class _Recorder:
    """Minimal FastMCP stand-in capturing registered tool names."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def add_tool(self, fn: Any) -> None:
        self.names.append(getattr(fn, "__name__", str(fn)))


def _registered(groups: str) -> set[str]:
    """Build a server with `groups`, capture its registration surface."""
    with patch("memini_ai.server.get_config") as cfg_mock:
        cfg_mock.return_value = MagicMock(tool_groups=groups)
        server = MCPServer()
    server._config = MagicMock(tool_groups=groups)  # noqa: SLF001
    recorder = _Recorder()
    server._mcp = recorder  # noqa: SLF001
    server._setup_tools()  # noqa: SLF001
    return set(recorder.names)


@pytest.mark.parametrize(
    ("groups", "expected_subset", "forbidden"),
    [
        (
            "core,trust,kanban,session",
            DEFAULT_EXPECTED,
            CHAIN_TRIO | REMOVED_CHAIN_TOOLS | KG_TOOLS,
        ),
        (
            "core,chains",
            CORE_ONLY | CHAIN_TRIO,
            REMOVED_CHAIN_TOOLS | KG_TOOLS | {"adjust_trust"},
        ),
        (
            "core,kg,dialectic",
            CORE_ONLY | KG_TOOLS | DIALECTIC_TOOLS,
            CHAIN_TRIO | {"adjust_trust"},
        ),
    ],
)
def test_group_profiles(
    groups: str, expected_subset: set[str], forbidden: set[str]
) -> None:
    names = _registered(groups)
    assert expected_subset <= names, f"missing: {sorted(expected_subset - names)}"
    assert not (names & forbidden), f"leaked: {sorted(names & forbidden)}"


def test_core_cannot_be_disabled() -> None:
    # Empty/garbage string still falls back so core memory tools survive.
    names = _registered("")
    assert {"query_memories", "add_memory"} <= names


def test_unknown_group_is_ignored() -> None:
    # Unknown names are skipped; known ones still take effect.
    names = _registered("core,bogus-group,trust")
    assert {"query_memories", "add_memory", "get_trust_score", "adjust_trust"} <= names
    assert not (names & CHAIN_TRIO)


def test_removed_chain_tools_never_register() -> None:
    all_names = _registered(
        "core,trust,kanban,session,chains,kg,dialectic,peers,memory_ops,audit,ops"
    )
    assert not (all_names & REMOVED_CHAIN_TOOLS)
