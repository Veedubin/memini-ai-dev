# memini-ai-dev

[![PyPI version](https://img.shields.io/pypi/v/memini-ai-dev.svg)](https://pypi.org/project/memini-ai-dev/)
[![CI](https://github.com/Veedubin/memini-ai-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/Veedubin/memini-ai-dev/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/memini-ai-dev.svg)](https://github.com/Veedubin/memini-ai-dev/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/memini-ai-dev.svg)](https://pypi.org/project/memini-ai-dev/)

> *"I remember"* in Latin (pronounced *meh-mee-nee*)

**Local-first semantic memory for AI agents.** Vector search, trust scoring,
knowledge graph, and persistent reasoning — fully MCP-compatible. Runs on
embedded PostgreSQL (no Docker) or an external Postgres + pgvector server.

## What it does

- **Remembers** facts, decisions, and patterns across sessions with
  trust-weighted retrieval
- **Answers** semantic queries with hybrid vector + BM25 search fused via RRF
- **Tracks** relationships between memories (SUPERSEDES, CONTRADICTS,
  DERIVED_FROM, RELATED_TO, PARTIAL_UPDATE)
- **Detects and resolves** contradictions via dialectic reasoning
- **Generates** project summaries (L0 ~100 tokens, L1 ~2K tokens) for session
  start
- **Indexes** codebases for semantic search across files
- **Shares** memories across agent peers with permission controls
- **Decays** low-trust memories and consolidates near-duplicates to keep the
  knowledge base relevant

## Architecture

```mermaid
flowchart LR
    U[User] -->|prompt| OC[OpenCode TUI]
    OC -->|task| ORCH["Neuralgentics Orchestrator<br/>12 personas + routing matrix"]
    ORCH -->|query / save| MEM[("memini-ai<br/>FIRST-CLASS MCP<br/>registered directly in opencode.json")]
    MEM --> PG[("PostgreSQL + pgvector<br/>trust-weighted memory")]
    ORCH -->|dispatch| AG["Specialist sub-agents<br/>coder · architect · tester · writer"]
    AG -->|"long-tail tool calls"| BRK["Neuralgentics Broker<br/>catalog · access control · audit"]
    BRK --> MCP["Brokered MCP servers<br/>searxng · github · videre · ssh<br/>behind the broker · on demand"]
    AG -->|outbound HTTP| GW["Neuralgentics Gateway<br/>egress policy + audit"]
    GW --> NET["Internet / LLM APIs"]
    MEM --> WEB["Neuralgentics Web<br/>dashboards"]
    GW --> WEB
    BRK --> WEB
```
memini-ai is a **first-class** MCP server — registered directly in `opencode.json` and always loaded. Every other MCP server sits **behind the broker**: catalog-advertised, access-controlled, and brokered on demand, which keeps long-tail tool schemas out of every prompt.


See the [memory lifecycle diagram](https://veedubin.github.io/memini-ai-dev/architecture/)
for details.

## Quickstart

### Zero-setup (embedded PostgreSQL)

```bash
uvx --from memini-ai-dev memini-ai --stdio
```

On v1.x the embedded `pgembed` backend self-bootstraps on first run — no Docker,
no external Postgres. Data lives under
`~/.local/share/memini-ai/pgembed/data` (XDG-compliant).

### External PostgreSQL

```bash
export MEMINI_DB_URL="postgresql://user:password@localhost:5432/memini"
export MEMINI_VECTOR_BACKEND="postgres-external"

uvx --from memini-ai-dev memini-ai --stdio
```

> **v1.0.0 breaking change:** if `MEMINI_DB_URL` is set you MUST also set
> `MEMINI_VECTOR_BACKEND=postgres-external`, otherwise the server raises
> `RuntimeError: memini-ai v1.0.0: MEMINI_DB_URL is set but
> MEMINI_VECTOR_BACKEND is not.` See [CHANGELOG.md](CHANGELOG.md) for the
> migration recipe.

#### TLS / SSL

memini-ai supports all six PostgreSQL `sslmode` values (`disable`, `allow`,
`prefer`, `require`, `verify-ca`, `verify-full`). The connection layer
(`src/memini_ai/postgres/database.py`) builds an `ssl.SSLContext` from the
`DB_SSLMODE` and `DB_SSLROOTCERT` environment variables (or the equivalent
fields on `MeminiConfig`).

For an external/team server you can configure TLS in two ways:

**Environment variables** (manual setup):

```bash
export MEMINI_DB_URL="postgresql://user:pw@db.host:5432/memini?sslmode=verify-full&sslrootcert=/etc/ssl/ca.pem"
export MEMINI_VECTOR_BACKEND="postgres-external"
export DB_SSLMODE="verify-full"
export DB_SSLROOTCERT="/etc/ssl/ca.pem"
```

**`memini-ai init` interactive flow** — when you choose a team/external
Postgres, the installer offers TLS:

```
memini-ai init --homedir --team
# → prompts for host/port/db/user/password, then:
#   SSL mode for the database connection:
#     1. prefer (default)   2. require   3. verify-ca
#     4. verify-full        5. disable
# → when verify-ca/verify-full is chosen, prompts for a CA cert path
# → validates the path exists and is PEM-encoded
# → writes sslmode + sslrootcert into the .env and DSN
# → runs a 5s connection test and reports success/failure (non-fatal)
```

**Flag shortcuts** (non-interactive, e.g. for CI or scripts):

| Flag | sslmode | sslrootcert |
|------|---------|-------------|
| `--tls` | `require` | (none) |
| `--tls-ca <path>` | `verify-full` | `<path>` (validated PEM) |
| `--no-tls` | `disable` | (none) |

```bash
# verify-full with a CA cert (recommended for production team servers):
memini-ai init --homedir --team --yes --tls-ca /etc/ssl/team-ca.pem

# require TLS without CA verification:
memini-ai init --homedir --team --yes --tls

# disable TLS (local socket / trusted network):
memini-ai init --homedir --team --yes --no-tls
```

The embedded `pgembed` backend (default) uses a local Unix socket and never
sets `sslmode` / `sslrootcert` — the TLS flags are ignored for the embedded
path.

### Minimal MCP client config (`opencode.json`)

If upgrading from v0.8.x with an existing external Postgres, set
`MEMINI_VECTOR_BACKEND=postgres-external` AND set `MEMINI_DB_URL`. v1.0.0+
defaults to `pgembed` (embedded) and refuses to start when `MEMINI_DB_URL`
is set without `MEMINI_VECTOR_BACKEND`.

```json
{
  "mcp": {
    "servers": {
      "memini-ai-dev": {
        "type": "local",
        "enabled": true,
        "environment": {
          // New installs (default): embedded PostgreSQL, no Docker.
          "MEMINI_VECTOR_BACKEND": "pgembed",
          // v0.8.x upgraders with an external server: uncomment BOTH lines.
          // "MEMINI_VECTOR_BACKEND": "postgres-external",
          // "MEMINI_DB_URL": "postgresql://postgres:password@localhost:5434/postgres",
          "TRUST_ENGINE": "true",
          "KG_ENABLED": "true"
        }
      }
    }
  }
}
```

## Features

### Memory Core

- **Embeddings**: MiniLM-L6-v2 (384-dim, default) or BGE-M3 (1024-dim, optional
  GPU upgrade). New deployments auto-upgrade to BGE-M3 when empty.
- **Search**: Hybrid vector + BM25 with Reciprocal Rank Fusion (RRF, k=60)
- **Multi-model RRF**: 384 + 1024 fusion in `auto` mode; optional 3rd CLIP
  image-recall arm when `MEMINI_IMAGE_SEARCH_ENABLED=true`
- **Project isolation**: Strict memory separation by `MEMINI_PROJECT_ID`

### Trust Engine

Trust starts at 0.5 and is adjusted via feedback signals:

| Signal            | Delta  |
|-------------------|--------|
| `agent_used`      | +0.05  |
| `user_confirmed`  | +0.10  |
| `agent_ignored`   | -0.02  |
| `user_corrected`  | -0.15  |

Archive threshold: < 0.2. Promote to L1: > 0.8. Temporal decay fades
irrelevant memories on a configurable half-life.

### Memory Graph

Relationships: SUPERSEDES, PARTIAL_UPDATE, RELATED_TO, CONTRADICTS,
DERIVED_FROM. Full supersession chain traversal (including archived memories).
Live D3.js force-directed visualization.

### Tiered Loading

- **L0 Summary** (~100 tokens): high-trust memories only (trust >= 0.5)
- **L1 Key Decisions** (~2K tokens): promoted memories (trust >= 0.8)
- **L2 Full Context**: all memories

### Knowledge Graph

Entity extraction from memory text, inference chains between entities, formal
KG queries with transitive closure, and HTML/D3.js graph export.

### Thought Chains

Persistent multi-step reasoning with branching, revisions, and abandonment.
API-compatible with sequential-thinking MCP servers.

### Dialectic

Contradiction detection across memories, dialectic resolution synthesis, and
per-memory argument history.

### Decay and Consolidation

Temporal trust decay keeps the knowledge base relevant; consolidation merges
near-duplicate memories on a configurable schedule.

### Multi-Peer Sharing

Peer registration with trust levels, per-memory sharing with permission levels
(shared, inherited, private), and opt-in per-project RBAC lockdown via
`MEMINI_PEER_ENFORCEMENT` + `MEMINI_PEER_ID`.

### MCP Tools (52)

See [docs/mcp-tools.md](docs/mcp-tools.md) for the full table. Categories:
Basic Memory, Project Indexing, Trust and Tiering, Thought Chains, Knowledge
Graph, Multi-Peer, User Modeling, System, and Advanced.

### Storage

- **Embedded PostgreSQL** (default, no Docker): `pgembed` (Postgres 17,
  in-process, multi-process shared)
- **External PostgreSQL**: any PostgreSQL 16+ with `pgvector` and
  `vectorscale` — set `MEMINI_VECTOR_BACKEND=postgres-external`
- **Optional team server RRF fusion**: `MEMINI_TEAM_DB_URL` +
  `MEMINI_FUSION_MODE=rrf`

### LLM Providers

- **Local Ollama** (default): any Ollama-compatible model
- **Ollama Cloud**: managed Ollama endpoints
- **OpenAI-compatible**: vLLM, LM Studio, and any OpenAI-compatible API

## Configuration

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_VECTOR_BACKEND` | `pgembed` | `pgembed` (embedded) or `postgres-external` |
| `MEMINI_DB_URL` | (unset) | PostgreSQL URL (external mode only) |
| `MEMINI_PGEMBED_DATA_DIR` | `~/.local/share/memini-ai/pgembed/data` | Embedded Postgres data dir |
| `MEMINI_TEAM_DB_URL` | (unset) | Optional team server URL for RRF fusion |
| `MEMINI_FUSION_MODE` | `none` | `none` or `rrf` (fuses embedded + team) |
| `MEMINI_EMBEDDING_DIM` | `384` | 384 or 1024 |
| `MEMINI_MODEL_NAME` | `all-MiniLM-L6-v2` | HF model ID or alias (`bge-m3`, `minilm`) |
| `MEMINI_EMBEDDING_MODE` | `auto` | `cpu`, `auto` (384 + 1024 RRF), `gpu` |
| `MEMINI_ENABLE_RRF` | `false` | Enable RRF fusion across MiniLM + BGE-M3 |
| `MEMINI_PEER_ENFORCEMENT` | `false` | Opt-in per-project RBAC lockdown |
| `MEMINI_PEER_ID` | (unset) | Project ID for RBAC tagging/filtering |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or OpenAI-compatible |
| `LLM_MODEL` | `llama3.2` | Model name passed to the LLM provider |
| `LLM_API_KEY` | (unset) | API key (optional for local Ollama) |
| `LLM_BASE_URL` | (unset) | Override base URL for OpenAI-compatible |

### Feature Toggles

| Feature | Env Var | Description |
|---------|---------|-------------|
| Trust Engine | `TRUST_ENGINE` | Trust scoring and archive/promotion |
| Tiered Loading | `TIERED_LOADING` | L0/L1/L2 summary generation |
| Knowledge Graph | `KG_ENABLED` | Entity extraction and KG queries |
| Memory Graph | `MEMORY_GRAPH` | Visual relationship mapping |
| Dialectic | `DIALECTIC_ENABLED` | Contradiction detection and resolution |
| Multi-Peer | `MULTI_PEER_ENABLED` | Peer-to-peer memory sharing |
| User Modeling | `USER_MODELING` | Persistent user profile tracking |
| Memory Decay | `DECAY_ENABLED` | Temporal trust decay engine |
| Auto-Extract | `AUTO_EXTRACT` | Auto extraction from conversations |
| Thought Chains | `THOUGHT_CHAINS` | Reasoning with branching/revision |
| Image Search | `MEMINI_IMAGE_SEARCH_ENABLED` | CLIP image-recall RRF arm |

See the [Configuration reference](https://veedubin.github.io/memini-ai-dev/configuration/)
for the full list.

## Documentation

- [Architecture](https://veedubin.github.io/memini-ai-dev/architecture/)
- [Configuration](https://veedubin.github.io/memini-ai-dev/configuration/)
- [MCP Tools](https://veedubin.github.io/memini-ai-dev/mcp-tools/)
- [Upgrading Embeddings](docs/upgrading-embeddings.md)
- [CHANGELOG](CHANGELOG.md)

## Development

```bash
uv sync          # install with dev dependencies
pytest           # run the test suite
ruff check src/  # lint
mypy src/        # type check
```

## License

MIT License — see [LICENSE](LICENSE) for details.