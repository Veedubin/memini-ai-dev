# memini-ai

> "I remember" in Latin (pronounced *meh-mee-nee*)

Local-first semantic memory server with vector search, trust scoring, and
persistent reasoning, fully MCP-compatible.

## Why memini-ai

Every long-running agent hits the same wall: the context window fills up, the
earlier half of the conversation is evicted, and the agent starts repeating
itself or forgetting decisions. memini-ai gives your agent a durable, queryable
memory backed by PostgreSQL + pgvector, with a trust engine so good memories
get reinforced and bad ones decay.

## Key features

- **MCP-compatible** - drop-in for OpenCode, Claude Desktop, or any MCP client
- **Trust engine** - every memory starts at trust 0.5 and is adjusted by agent
  and user feedback (`agent_used` +0.05, `user_confirmed` +0.10, etc.)
- **Tiered memory** - L0 (~100 tokens, summary), L1 (~2K, key decisions),
  L2 (full context) for efficient context loading
- **Knowledge graph** - entity extraction, typed relationships, live D3.js
  visualization, inference chains between entities
- **Dialectic reasoning** - automatic contradiction detection and LLM-driven
  resolution between conflicting memories
- **Memory decay** - temporal trust decay keeps the store focused on what
  still matters
- **Multi-peer** - share memory subsets across distinct agent personas
- **Dual-model RRF** - 384-dim MiniLM (CPU default) plus optional 1024-dim
  BGE-M3 (GPU), fused with Reciprocal Rank Fusion (k=60)
- **Persistent thought chains** - branching, revisable reasoning chains that
  survive across sessions
- **Project isolation** - strict per-project memory separation via `project_id`

## Install

```bash
# Recommended: run via uvx (no install needed)
uvx --from memini-ai-dev memini-ai --stdio

# Or install with pip
pip install memini-ai-dev
memini-ai --stdio
```

memini-ai v1.x self-bootstraps an embedded PostgreSQL 17 + pgvector server on
first run - no Docker, no external database required. The data directory lives
at `~/.local/share/memini-ai/pgembed/data` by default.

## First MCP call

Point any MCP client at the `memini-ai --stdio` command and call:

```json
{
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": { "content": "memini-ai is now installed" }
  }
}
```

Then retrieve it:

```json
{
  "method": "tools/call",
  "params": {
    "name": "query_memories",
    "arguments": { "query": "install status" }
  }
}
```

## Next steps

- [Getting started](getting-started.md) - full install, bootstrap, and config
- [Configuration](configuration.md) - every env var and feature toggle
- [MCP tools](mcp-tools.md) - all 52 tools grouped by category
- [Architecture](architecture.md) - components and memory lifecycle
- [Upgrading embeddings](upgrading-embeddings.md) - MiniLM to BGE-M3 migration
- [Changelog](changelog.md) - per-release history

## Links

- [PyPI](https://pypi.org/project/memini-ai-dev/)
- [GitHub](https://github.com/Veedubin/memini-ai-dev)
- [pgvector](https://github.com/pgvector/pgvector)
- [FastMCP](https://github.com/jlowin/fastmcp)