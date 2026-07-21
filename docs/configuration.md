# Configuration

memini-ai is configured via environment variables (or a JSON config file at
`MEMINI_CONFIG_PATH`). All settings are optional - defaults work for a solo
local-first install.

## Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_VECTOR_BACKEND` | `pgembed` | `pgembed` (in-process Postgres 17) or `postgres-external` (legacy v0.8.x Docker/team server). **Required if `MEMINI_DB_URL` is set.** |
| `MEMINI_DB_URL` | (empty) | PostgreSQL connection URL for external mode. Ignored when `MEMINI_VECTOR_BACKEND=pgembed`. |
| `MEMINI_PGEMBED_DATA_DIR` | `~/.local/share/memini-ai/pgembed/data` | Data directory for the embedded Postgres server. |
| `MEMINI_TEAM_DB_URL` | (empty) | Optional team server URL for RRF fusion (writes go to primary only). |
| `MEMINI_FUSION_MODE` | (empty) | Set to `rrf` when `MEMINI_TEAM_DB_URL` is populated to fuse reads across primary + team. |
| `MEMINI_PROJECT_ID` | auto-generated | Project identifier for memory isolation. |
| `MEMINI_EMBEDDING_DIM` | `384` | Embedding dimension (384 for MiniLM, 1024 for BGE-M3). |
| `MEMINI_MODEL_NAME` | `all-MiniLM-L6-v2` | Active embedding model. Accepts short aliases (`minilm`, `bge-m3`) or full HF names. |
| `MEMINI_EMBEDDING_MODE` | `auto` | `cpu` (384-only), `auto` (384 writes + RRF queries across 384 and 1024), `gpu` (1024-only). |
| `MEMINI_ENABLE_RRF` | `false` | Enable RRF fusion across MiniLM + BGE-M3 columns. |
| `RRF_TOP_K_PER_MODEL` | `20` | Per-model top-k before RRF fusion. |
| `RRF_K` | `60` | Reciprocal Rank Fusion k constant (Cormack SIGIR 2009). |
| `MEMINI_ENABLED_MODELS` | `["all-MiniLM-L6-v2", "BAAI/bge-m3"]` | Models used for RRF dispatch. |
| `MEMINI_AUTO_DETECT_MODEL` | `true` | New deployments with 0 memories auto-upgrade to BGE-M3 (1024-dim). |
| `MEMINI_STRICT_EMBEDDING_DIM` | `false` | Dim mismatch raises `RuntimeError` instead of degrading to text-only. |
| `MEMINI_ELEVATE_ENABLED` | `true` | Enable the `elevate_memory_to_1024` MCP tool. |
| `MEMINI_CHUNK_SIZE` | `512` | Chunk size for project file indexing. |
| `MEMINI_CHUNK_OVERLAP` | `50` | Overlap between chunks. |
| `MEMINI_BATCH_SIZE` | `32` | Batch size for embedding generation. |
| `MEMINI_WORKERS` | cpu_count | Number of worker threads. |
| `MEMINI_LOG_LEVEL` | `info` | Logging level (`debug`, `info`, `warning`, `error`). |
| `MEMINI_DEVICE` | `auto` | Device for embeddings (`auto`, `cpu`, `cuda`). |
| `MEMINI_CONFIG_PATH` | (empty) | Path to a JSON config file. |

## Feature toggles

All features below are **disabled by default**. Set to `true` to enable.

| Feature | Env var | What it does |
|---------|---------|--------------|
| Trust engine | `MEMINI_TRUST_ENGINE` | Trust scoring with archive/promotion logic. |
| Tiered loading | `MEMINI_TIERED_LOADING` | L0/L1/L2 summary generation for efficient context loading. |
| Knowledge graph | `MEMINI_KG_ENABLED` | Entity extraction and KG queries. |
| Memory graph | `MEMINI_MEMORY_GRAPH` | Visual relationship mapping. |
| Dialectic | `MEMINI_DIALECTIC_ENABLED` | Contradiction detection and resolution. |
| Multi-peer | `MEMINI_MULTI_PEER_ENABLED` | Peer-to-peer memory sharing. |
| User modeling | `MEMINI_USER_MODELING` | Persistent user profile and style tracking. |
| Memory decay | `MEMINI_DECAY_ENABLED` | Temporal trust decay engine. |
| Auto-extract | `MEMINI_AUTO_EXTRACT` | Automatic memory extraction from conversations. |
| Pre-compression | `MEMINI_PRECOMPRESS` | Context-aware pre-compression extraction. |
| Thought chains | `THOUGHT_CHAINS` | Persistent reasoning with branching/revision. |

## Trust engine tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_TRUST_THRESHOLD_ARCHIVE` | `0.2` | Archive memories below this trust score. |
| `MEMINI_TRUST_THRESHOLD_PROMOTE` | `0.8` | Promote to L1 above this trust score. |
| `MEMINI_TRUST_DELTA_USE` | `+0.05` | Trust delta for `agent_used` signal. |
| `MEMINI_TRUST_DELTA_IGNORED` | `-0.05` | Trust delta for `agent_ignored` signal. |
| `MEMINI_TRUST_DELTA_CORRECT` | `-0.15` | Trust delta for `user_corrected` signal. |
| `MEMINI_TRUST_DELTA_CONFIRM` | `+0.10` | Trust delta for `user_confirmed` signal. |

## Image recall (v0.8.0+)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_IMAGE_SEARCH_ENABLED` | `false` | Master gate for the image-recall RRF arm. |
| `MEMINI_IMAGE_CLIP_MODEL` | `clip-ViT-B-32` | CLIP model (`clip-ViT-B-32` or `clip-ViT-L-14`). |
| `MEMINI_IMAGE_CLIP_DEVICE` | `auto` | Device (`auto`, `cpu`, `cuda`). |
| `MEMINI_IMAGE_DIR` | `~/.memini-ai/images` | Filesystem directory for stored images. |
| `MEMINI_IMAGE_DB_URL` | (empty, falls back to `MEMINI_DB_URL`) | PostgreSQL URL for the image index. |

## TLS / SSL

PostgreSQL connections support TLS encryption to prevent data exfiltration.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_SSLMODE` | `prefer` | PostgreSQL SSL mode. |
| `DB_SSLROOTCERT` | (empty) | Path to CA certificate for server verification. |

SSL mode values (from the [libpq docs](https://www.postgresql.org/docs/current/libpq-ssl.html)):

| Value | Encryption | Server cert verified | Hostname verified | Use case |
|-------|-----------|---------------------|-------------------|----------|
| `disable` | No | No | No | Development only (no TLS) |
| `allow` | Optional | No | No | Rarely useful |
| `prefer` | Tried first | No | No | Default - fallback to plaintext |
| `require` | Yes | No | No | Encrypted but no identity check |
| `verify-ca` | Yes | Yes | No | CA verified, hostname not checked |
| `verify-full` | Yes | Yes | Yes | Recommended for production |

## RBAC: per-project users

By default all users have read/write/edit access to all memories (open
access). Per-project isolation is opt-in.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_PEER_ENFORCEMENT` | `false` | When `true`, queries filter by `peer_id`. |
| `MEMINI_PEER_ID` | (empty) | The project name to filter on (when enforcement is on). |

User management CLI:

```bash
memini-ai user add --name <name> [--admin]    # Add a user
memini-ai user list                            # List all users
memini-ai user remove --name <name>            # Drop a user
```

## The .env file pattern

- **Homedir `.env`** (`~/.config/opencode/.env`): admin credentials.
- **Project `.env`** (`./.env`): project credentials.
- `opencode.json` references via `{env:MEMINI_PROJECT_USER}` etc. - secrets
  are never committed to git.
- `.env` files are chmod 600.

## Quick start with all features on

```bash
export MEMINI_TRUST_ENGINE=true
export MEMINI_TIERED_LOADING=true
export MEMINI_KG_ENABLED=true
export MEMINI_DIALECTIC_ENABLED=true
export MEMINI_DECAY_ENABLED=true
export THOUGHT_CHAINS=true
export MEMINI_MULTI_PEER_ENABLED=true
uvx --from memini-ai-dev memini-ai --stdio
```