# Getting started

## Prerequisites

- Python 3.12+ (pgembed 0.2.0 requires 3.12+)
- ~1 GB free disk for the embedding model + pgembed data dir
- No Docker required (v1.x self-bootstraps an embedded Postgres)

## Install

=== "uvx (recommended)"

    No install step - run directly:

    ```bash
    uvx --from memini-ai-dev memini-ai --stdio
    ```

=== "pip"

    ```bash
    pip install memini-ai-dev
    memini-ai --stdio
    ```

=== "Source"

    ```bash
    git clone https://github.com/Veedubin/memini-ai-dev.git
    cd memini-ai-dev
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    memini-ai --stdio
    ```

## Bootstrap the database

### Default: embedded pgembed (v1.x)

memini-ai v1.x starts an in-process PostgreSQL 17 server on first run. The
data directory lives at `~/.local/share/memini-ai/pgembed/data` (XDG-compliant).
No Docker, no external database, no manual `CREATE EXTENSION` calls.

```bash
# One-time: initialize the embedded server + write opencode.json
memini-ai init

# Or just start the MCP server - the embedded backend self-bootstraps
memini-ai --stdio
```

### Optional: external PostgreSQL + pgvector

Use this if you already run PostgreSQL, want a team server, or need Docker.

```bash
# Point at an external PostgreSQL 16+ with pgvector + vectorscale
export MEMINI_VECTOR_BACKEND=postgres-external
export MEMINI_DB_URL=postgresql://user:pass@host:5432/dbname
memini-ai --stdio
```

!!! warning "v1.0.0 breaking change"
    If you previously ran v0.8.x with `MEMINI_DB_URL` set, you MUST add
    `MEMINI_VECTOR_BACKEND=postgres-external` or the server will refuse to
    start. This preserves v0.8.x behavior exactly - no data migration needed.

### Migrate from v0.8.x to embedded

```bash
# Copy external Postgres data into the embedded server
memini-ai migrate --from='postgresql://user:pass@host:5432/dbname'

# Dry-run first to sanity-check source row counts
memini-ai migrate --dry-run --from='postgresql://user:pass@host:5432/dbname'
```

The migrate command pre-installs the `vector` and `vectorscale` extensions
on the target, excludes `timescaledb` from the dump, verifies per-table row
counts after restore, and exits non-zero on mismatch. Source DB is untouched.

## Configure your MCP client

=== "OpenCode"

    Add to `.opencode/opencode.json`:

    ```json
    {
      "mcp": {
        "memini-ai-dev": {
          "type": "local",
          "command": ["uvx", "--from", "memini-ai-dev", "memini-ai", "--stdio"],
          "enabled": true
        }
      }
    }
    ```

    Or use the init CLI: `memini-ai init --homedir` (writes the global config)
    and `memini-ai init --project` (writes a project-local config).

=== "Claude Desktop"

    Add to your `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "memini-ai-dev": {
          "command": "uvx",
          "args": ["--from", "memini-ai-dev", "memini-ai", "--stdio"]
        }
      }
    }
    ```

## First MCP call

Once your client connects, try a write + read round-trip:

```json
{
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": { "content": "memini-ai is now installed and working" }
  }
}
```

```json
{
  "method": "tools/call",
  "params": {
    "name": "query_memories",
    "arguments": { "query": "install status" }
  }
}
```

If both return without error, you are up and running. The `healthcheck` tool
does a write + read round-trip and reports latency:

```json
{
  "method": "tools/call",
  "params": { "name": "healthcheck", "arguments": {} }
}
```

## Next steps

- [Configuration](configuration.md) - all env vars and feature toggles
- [MCP tools](mcp-tools.md) - the full tool surface
- [Architecture](architecture.md) - how the pieces fit
- [Upgrading embeddings](upgrading-embeddings.md) - MiniLM to BGE-M3 migration