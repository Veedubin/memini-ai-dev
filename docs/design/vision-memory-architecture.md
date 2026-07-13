# Vision Memory Architecture for memini-ai (v2 — Shared Package Model)

**Date:** 2026-07-13
**Status:** DESIGN COMPLETE — All 8 user decisions incorporated. Ready for implementation.
**Supersedes:** v1 design (2026-07-12) — Sections 1, 5, 7, 8 of the prior doc are replaced by this v2.
**Target versions:** memini-vision 0.1.0, videre-mcp 0.3.0, memini-ai 0.8.0

## 1. Overview

Vision memory is implemented as a **three-repo split** with a shared Python library at the center. videre-mcp exposes the new vision MCP tools (it is already the agent's vision front door). memini-ai uses the same shared library internally to fuse image results into its existing Reciprocal Rank Fusion (RRF) pipeline. A single env var — `MEMINI_IMAGE_SEARCH_ENABLED` — gates the entire subsystem end-to-end. When `false` (the default), text-only users see zero behavior change.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent (boomerang-coder)                      │
│                                                                     │
│  "take a screenshot"          "query_memories('auth bug')"          │
│        │                              │                             │
│        ▼                              ▼                             │
│  ┌──────────────┐            ┌──────────────────┐                   │
│  │  videre-mcp  │            │   memini-ai-dev  │                   │
│  │  (v0.3.0)    │            │   (v0.8.0)       │                   │
│  │              │            │                  │                   │
│  │ 6 existing   │            │ query_memories() │                   │
│  │ tools + 3    │            │   ┌─ 384 RRF ──┐ │                   │
│  │ new vision   │            │   ├─ 1024 RRF ─┤ │                   │
│  │ tools        │            │   └─ CLIP RRF ─┘ │ (when enabled)   │
│  └──────┬───────┘            └────────┬─────────┘                   │
│         │                             │                             │
│         └──────────┬──────────────────┘                             │
│                    ▼                                                │
│         ┌─────────────────────┐                                     │
│         │   memini-vision     │  ← shared library (PyPI package)    │
│         │   (v0.1.0)          │                                     │
│         │                     │                                     │
│         │  ImageStore         │  → ~/.memini-ai/images/             │
│         │  ClipEmbedder       │  → sentence-transformers CLIP       │
│         │  ImageIndex         │  → PostgreSQL memories_image table  │
│         │  ImageQuery         │  → cross-modal CLIP search          │
│         │  VisionConfig       │  → pydantic-settings env vars       │
│         └─────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle:** videre-mcp is the agent's natural front door for vision (it already owns screenshot/OCR/describe). memini-ai is the agent's natural back end for unified RRF (its existing `query_memories` is what agents call). `memini-vision` is the single source of truth for storage and embedding logic — no duplication.

## 2. The `memini-vision` Shared Package

A new standalone Python package at `/home/jcharles/Projects/python/memini-vision/`, versioned independently, published to PyPI as `memini-vision`. It is a **library**, not an MCP server.

### 2.1 Module Structure

```
memini-vision/
├── pyproject.toml
├── src/memini_vision/
│   ├── __init__.py
│   ├── config.py          # VisionConfig (pydantic-settings)
│   ├── store.py           # ImageStore (filesystem pointer storage)
│   ├── embedder.py        # ClipEmbedder (sentence-transformers CLIP)
│   ├── index.py           # ImageIndex (PostgreSQL CRUD for memories_image)
│   ├── query.py           # ImageQuery (cross-modal CLIP search)
│   └── types.py           # Shared types (ImageRecord, SearchResult, etc.)
└── tests/
```

### 2.2 Public API Surface

| Class | Responsibility | Key Methods |
|-------|---------------|-------------|
| `VisionConfig` | pydantic-settings for `MEMINI_IMAGE_*` env vars | — |
| `ImageStore` | Filesystem storage with sha256 sharding | `store(image_bytes, mime_type) → ImageRecord`, `get_path(sha256) → Path`, `delete(sha256)` |
| `ClipEmbedder` | Lazy-loaded CLIP model via sentence-transformers | `encode_image(pil_image) → list[float]`, `encode_text(text) → list[float]`, `model_dim → int` |
| `ImageIndex` | PostgreSQL CRUD for `memories_image` table | `insert(record) → uuid`, `get_by_id(uuid) → ImageRecord`, `get_by_memory_id(uuid) → ImageRecord`, `delete(uuid)` |
| `ImageQuery` | Cross-modal vector search | `search_by_text(query_text, limit) → list[SearchResult]`, `search_by_image(pil_image, limit) → list[SearchResult]` |

### 2.3 VisionConfig (pydantic-settings)

```python
class VisionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMINI_IMAGE_")

    search_enabled: bool = False       # MEMINI_IMAGE_SEARCH_ENABLED
    clip_model: str = "clip-ViT-B-32" # MEMINI_IMAGE_CLIP_MODEL
    clip_device: str = "auto"          # MEMINI_IMAGE_CLIP_DEVICE
    image_dir: str = "~/.memini-ai/images"  # MEMINI_IMAGE_DIR
    max_file_size: int = 10 * 1024 * 1024   # MEMINI_IMAGE_MAX_FILE_SIZE
    db_url: str = ""                   # MEMINI_IMAGE_DB_URL (falls back to MEMINI_DB_URL)
```

### 2.4 What `memini-vision` Does NOT Do

- **No FastMCP server.** It is a library, not an MCP server. No `@mcp.tool()` decorators, no `mcp.run()`.
- **No CLI entry point.** No `console_scripts` in pyproject.toml.
- **No schema migration ownership.** The `memories_image` table DDL lives in memini-ai's migration system (`migrations/postgres/000008_*.sql`). `memini-vision`'s `ImageIndex` assumes the table exists — it is the caller's responsibility to ensure the schema is applied.
- **No text embedding.** CLIP only. Text embeddings (MiniLM, BGE-M3) remain memini-ai's domain.

## 3. videre-mcp v0.3.0 Changes

videre-mcp lives at `/home/jcharles/Projects/python/videre-mcp/`. Its 6 existing tools (`take_screenshot`, `describe_image`, `describe_screenshot`, `ocr_image`, `ocr_paddle`, `parse_document`) in `src/videre_mcp/server.py` (733 lines) are unchanged.

### 3.1 Three New MCP Tools

All three tools are registered on the existing `mcp = FastMCP("videre-mcp")` instance (server.py:160). They return **absolute filesystem paths only** — no base64 encoding.

#### `save_image_memory`

```
Signature:
  save_image_memory(
      image_path: str,              # Path to image file on disk (PNG/JPEG)
      text: str,                    # Memory text (required — the context)
      caption: str | None = None,   # Optional human-readable caption
      tags: list[str] | None = None,# Optional tags for filtering
      source_type: str = "session", # Memory source type
  ) -> dict:
      memory_id: str,               # UUID of the created memories row
      image_id: str,                # UUID of the created memories_image row
      sha256: str,                  # SHA-256 of the image bytes
      file_path: str,              # Absolute path where the image was stored
```

**What it does:**
1. Reads image bytes from `image_path`.
2. Calls `ImageStore.store()` → copies to `{MEMINI_IMAGE_DIR}/{sha256[:2]}/{sha256}.{ext}`, returns `ImageRecord`.
3. Calls `ClipEmbedder.encode_image()` → CLIP vector.
4. Calls `ImageIndex.insert()` → writes to `memories_image` table.
5. Creates a `memories` row (via memini-ai's existing `add_memory` MCP tool, or directly via asyncpg if videre-mcp has DB access — **implementation detail TBD**).
6. Returns `{memory_id, image_id, sha256, file_path}`.

#### `query_images`

```
Signature:
  query_images(
      query: str,                   # Text query (cross-modal: CLIP text tower)
      limit: int = 10,              # Max results
  ) -> list[dict]:
      memory_id: str,
      text: str,                    # The associated memory text
      image_id: str,
      file_path: str,               # Absolute filesystem path
      sha256: str,
      caption: str | None,
      distance: float,              # Cosine distance (lower = more similar)
```

**What it does:**
1. Calls `ImageQuery.search_by_text(query, limit)`.
2. Returns ranked list with image metadata + associated memory text. **File paths only — no base64.**

#### `get_image`

```
Signature:
  get_image(
      image_id: str | None = None,  # UUID of memories_image row
      memory_id: str | None = None, # Or look up by memory UUID
  ) -> dict:
      image_id: str,
      memory_id: str,
      file_path: str,               # Absolute filesystem path
      mime_type: str,
      width: int | None,
      height: int | None,
      file_size_bytes: int,
      sha256: str,
      caption: str | None,
```

**What it does:**
1. Calls `ImageIndex.get_by_id()` or `ImageIndex.get_by_memory_id()`.
2. Returns metadata + the absolute filesystem path. **No base64, no `format` parameter.**

### 3.2 pyproject.toml Changes

Add a new optional dependency group in `pyproject.toml` (following the existing pattern at lines 41-52):

```toml
[project.optional-dependencies]
vision = ["memini-vision>=0.1.0"]
```

The existing groups (`dev`, `deep`, `paddle`, `docling`, `optimize`) are unchanged.

### 3.3 Tool-Disable Pattern: Conditional `@mcp.tool()` Registration

**Decision: Conditional registration at module load time.** The three new tools are wrapped in an `if VisionConfig().search_enabled:` block at module level. When `MEMINI_IMAGE_SEARCH_ENABLED=false` (the default), the `@mcp.tool()` decorators are never called, and the tools do not appear in the MCP tool list.

**Why conditional registration over per-call checks:**
- Cleaner: the tools simply don't exist when disabled. No "this tool is disabled" error messages to handle.
- Matches the lifecycle model: flipping the flag is like toggling an MCP server in `opencode.json`.
- The existing 6 tools are unaffected — they are registered unconditionally above the guard.
- FastMCP's tool list is built at import time, so conditional registration is the natural pattern.

**Implementation sketch (in `server.py`, after line 160):**

```python
from memini_vision.config import VisionConfig

_vision_config = VisionConfig()  # reads MEMINI_IMAGE_SEARCH_ENABLED

# ... existing 6 tools registered unconditionally ...

if _vision_config.search_enabled:
    @mcp.tool()
    def save_image_memory(...): ...

    @mcp.tool()
    def query_images(...): ...

    @mcp.tool()
    def get_image(...): ...
```

### 3.4 Version Bump: 0.2.1 → 0.3.0

Minor version bump. New feature (3 tools), backwards compatible (existing 6 tools unchanged). The `vision` extra is optional — users who don't install it see no change.

## 4. memini-ai v0.8.0 Changes

memini-ai lives at `/home/jcharles/Projects/MCP-Servers/memini-ai-dev/`. It does **NOT** expose the 3 vision tools on its own MCP surface — videre-mcp does that. memini-ai only uses `memini-vision` **internally** to wire image results into its existing RRF pipeline.

### 4.1 New Environment Variables

All use the `MEMINI_IMAGE_` prefix, consumed via `memini-vision.VisionConfig`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_IMAGE_SEARCH_ENABLED` | `false` | Master gate. When `false`, no CLIP model loads, no image table is queried, RRF stays 2-list. |
| `MEMINI_IMAGE_CLIP_MODEL` | `clip-ViT-B-32` | CLIP model ID. `clip-ViT-B-32` (512-dim, ~400MB RAM) or `clip-ViT-L-14` (768-dim, ~1.5GB RAM). |
| `MEMINI_IMAGE_CLIP_DEVICE` | `auto` | Device for CLIP. `auto` (CUDA if available, else CPU), `cpu`, or `cuda`. |
| `MEMINI_IMAGE_DIR` | `~/.memini-ai/images` | Filesystem directory for stored images. Sharded by first 2 hex chars of SHA-256. |
| `MEMINI_IMAGE_MAX_FILE_SIZE` | `10485760` (10MB) | Maximum image file size in bytes. |
| `MEMINI_IMAGE_DB_URL` | (empty — falls back to `MEMINI_DB_URL`) | PostgreSQL connection string for the image index. |

### 4.2 pyproject.toml Changes

Add a new optional dependency group:

```toml
[project.optional-dependencies]
vision = ["memini-vision>=0.1.0"]
```

### 4.3 Schema Migration

**Migration file:** `src/memini_ai/postgres/migrations/000008_add_memories_image.up.sql`

Following the `memories_1024` precedent (`schema.py:117-174`):

```sql
CREATE TABLE IF NOT EXISTS memories_image (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    sha256 VARCHAR(64) UNIQUE NOT NULL,
    mime_type VARCHAR(50) NOT NULL DEFAULT 'image/png',
    width INT,
    height INT,
    file_size_bytes BIGINT,
    embedding vector(768) NOT NULL,       -- 768-dim to accommodate both CLIP models
    caption TEXT,
    caption_embedding vector(384),         -- MiniLM embedding of caption
    source_tool VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding_model VARCHAR(100) DEFAULT 'clip-ViT-B-32'
);

CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
USING diskann (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memories_image_memory_id ON memories_image(memory_id);
CREATE INDEX IF NOT EXISTS idx_memories_image_sha256 ON memories_image(sha256);
CREATE INDEX IF NOT EXISTS idx_memories_image_created_at ON memories_image(created_at DESC);
```

**Rollback:** `src/memini_ai/postgres/migrations/000008_add_memories_image.down.sql`:

```sql
DROP TABLE IF EXISTS memories_image CASCADE;
```

**Key design decisions:**
- `vector(768)` accommodates both ViT-B/32 (512-dim, zero-padded) and ViT-L/14 (768-dim, native). No separate `image_embedding_dim` config field — dim is derived from the model.
- `memory_id UUID NOT NULL UNIQUE` — one image per memory (1:1). If a memory needs multiple images, create multiple memories.
- `ON DELETE CASCADE` — if the text memory is hard-deleted, the image row goes with it.
- The table is created at memini-ai startup **regardless** of whether `MEMINI_IMAGE_SEARCH_ENABLED` is `true`. This is idempotent (`IF NOT EXISTS`) and ensures videre-mcp and memini-ai can both write to it without coordination problems.

### 4.4 `source_type` CHECK Extension

The `memories` table's `source_type` CHECK constraint (`schema.py:52-54`) is extended to include `'image'`:

```sql
-- Before (v0.7.9):
source_type IN ('session', 'file', 'web', 'boomerang', 'project')

-- After (v0.8.0):
source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'image')
```

This is applied via `ALTER TABLE` in the migration. Existing rows are unaffected.

### 4.5 RRF Integration: Third Fan-Out Arm

The existing `_query_dual_model_rrf` method (`memory/system.py:436-517`) fans out to two ranked lists (384-dim MiniLM + 1024-dim BGE-M3) and fuses with `reciprocal_rank_fusion()` (`rrf.py:28-93`) using k=60. When `MEMINI_IMAGE_SEARCH_ENABLED=true`, a **third ranked list** is added.

**Modification to `_query_dual_model_rrf` (system.py:436):**

```python
async def _query_dual_model_rrf(self, question, options):
    # ... existing 384 + 1024 fan-out (lines 456-498) ...

    # NEW: Image fan-out (only when enabled)
    image_ids: list[str] = []
    if _vision_config.search_enabled:
        try:
            from memini_vision.query import ImageQuery
            image_results = await ImageQuery(self._db.pool).search_by_text(
                question, limit=fetch_k
            )
            image_ids = [r.memory_id for r in image_results]
        except Exception:
            logger.warning("Image search failed, skipping image RRF arm")

    # RRF fusion over 2 or 3 lists
    ranked_ids = [
        [e.id for e in results_384],
        [e.id for e in results_1024],
    ]
    if image_ids:
        ranked_ids.append(image_ids)

    fused_ids = rrf_with_limit(ranked_ids, k=self._resolved_rrf_k, limit=options.top_k)
    # ... re-hydrate (lines 509-517) ...
```

**No change to `reciprocal_rank_fusion()` itself** — it already accepts `Sequence[Sequence[T]]` (any number of lists). The RRF math is:

```
For each memory ID that appears in any list:
  rrf_score = 0
  if in text_384_list at rank r1:  rrf_score += 1/(k + r1)
  if in text_1024_list at rank r2: rrf_score += 1/(k + r2)
  if in image_list at rank r3:     rrf_score += 1/(k + r3)
```

A memory with both a text match AND an image match gets both contributions summed — the natural boost for multi-modal agreement.

**When `MEMINI_IMAGE_SEARCH_ENABLED=false` (the default):** The `if _vision_config.search_enabled:` block is skipped entirely. The query path is byte-for-byte identical to v0.7.9. No CLIP model is loaded. No image table is queried.

### 4.6 Table Creation at Startup

The `memories_image` table is created at memini-ai startup via the idempotent migration, **regardless** of whether image search is enabled. This follows the same pattern as `memories_1024` (v0.7.0): the table exists but is empty and unqueried until the feature is enabled. This ensures videre-mcp can write to the table (via `memini-vision.ImageIndex`) without memini-ai needing image search enabled.

### 4.7 Version Bump: 0.7.9 → 0.8.0

Minor version bump. Additive feature (new table, new RRF arm), backwards compatible (text-only users see zero change). The `vision` extra is optional.

## 5. Agent Workflows

### 5.1 Agent Takes a Screenshot and Saves It

```
Agent: videre-mcp_take_screenshot(monitor=1)
  → returns {"path": "/tmp/screenshot_20260713_120000.png", "width": 1920, "height": 1080}

Agent: videre-mcp_save_image_memory(
    image_path="/tmp/screenshot_20260713_120000.png",
    text="Terminal showing the auth bug reproduction — KeyError on line 342",
    caption="Screenshot of terminal with Python traceback for auth bug",
    tags=["auth", "bug", "traceback"],
    source_type="session"
  )
  → returns {"memory_id": "abc-123", "image_id": "def-456", "sha256": "a1b2...", "file_path": "/home/user/.memini-ai/images/a1/a1b2c3...png"}
```

### 5.2 Agent Recalls a Screenshot from Text

```
Agent: videre-mcp_query_images("terminal showing a Python traceback", limit=5)
  → returns [
      {"memory_id": "abc-123", "text": "Terminal showing the auth bug...", "file_path": "/home/user/.memini-ai/images/a1/a1b2c3...png", "distance": 0.23},
      ...
    ]

Agent opens the returned file_path to view the screenshot.
```

### 5.3 Agent Gets Unified RRF Results (Text + Image)

```
Agent: memini-ai-dev_query_memories("debugging the auth bug", limit=10)
  → returns ranked list containing BOTH:
      - Text memories about the auth bug (from MiniLM + BGE-M3 RRF)
      - Image memories of screenshots showing the auth bug (from CLIP RRF)
    All fused into a single ranked list by RRF k=60.
```

A memory that appears in both text and image lists gets a higher fused score — the natural boost for multi-modal agreement.

### 5.4 Agent Describes an Image (Existing Tool, No Change)

```
Agent: videre-mcp_describe_image("/tmp/screenshot.png", detail_level="high")
  → returns {"description": "A terminal window showing a Python traceback with KeyError on line 342...", "model": "Florence-2-base"}
```

This is one of the 6 existing tools (`server.py:274-311`). No change in v0.3.0.

### 5.5 User Toggles Image Search Off

```
1. Set MEMINI_IMAGE_SEARCH_ENABLED=false in .env or opencode.json env block
2. Restart videre-mcp and memini-ai-dev MCP servers
3. Result:
   - save_image_memory, query_images, get_image disappear from videre-mcp's tool list
   - memini-ai-dev query_memories stops fanning out to the image table
   - CLIP model is never loaded
   - Text-only behavior is identical to v0.7.9
```

## 6. Lifecycle / On-Off Behavior

### 6.1 memini-ai Server Start (MEMINI_IMAGE_SEARCH_ENABLED=false — default)

1. Migration 000008 runs → `memories_image` table created (idempotent, empty).
2. `source_type` CHECK extended to include `'image'`.
3. `VisionConfig.search_enabled` is `False` → `_query_dual_model_rrf` skips the image fan-out arm.
4. No CLIP model is loaded. No `memini-vision` import occurs (lazy import inside the `if` block).
5. `query_memories` behavior is byte-for-byte identical to v0.7.9.

### 6.2 memini-ai Server Start (MEMINI_IMAGE_SEARCH_ENABLED=true)

1. Same as above, plus:
2. On first `query_memories` call, `memini-vision` is imported lazily.
3. `ClipEmbedder` loads the CLIP model on first use (lazy loading — not at server start).
4. The image fan-out arm activates in `_query_dual_model_rrf`.
5. RRF fuses 3 lists instead of 2.

### 6.3 videre-mcp Server Start (MEMINI_IMAGE_SEARCH_ENABLED=false — default)

1. `VisionConfig().search_enabled` is `False` → the `if` block at module level is skipped.
2. The 3 `@mcp.tool()` decorators are never called.
3. The MCP tool list contains only the 6 existing tools.
4. No `memini-vision` import occurs.

### 6.4 videre-mcp Server Start (MEMINI_IMAGE_SEARCH_ENABLED=true)

1. `VisionConfig().search_enabled` is `True` → the `if` block executes.
2. The 3 `@mcp.tool()` decorators register the new tools.
3. `memini-vision` is imported at module load time.
4. `ClipEmbedder` loads the CLIP model **lazily on first use** (first `save_image_memory` or `query_images` call), not at server start.

### 6.5 Lazy CLIP Model Load

CLIP models are large (ViT-B/32: ~150MB download, ~400MB RAM; ViT-L/14: ~890MB download, ~1.5GB RAM). They load **lazily on first use**, not at server start:

```python
class ClipEmbedder:
    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model
```

This means:
- Server startup is fast (no model download/load).
- First `save_image_memory` or `query_images` call incurs a 2-5s (B/32) or 5-10s (L/14) latency for model loading.
- Subsequent calls use the cached model (sub-millisecond overhead).

### 6.6 Verifying Image Search Is Off

- **videre-mcp:** List the MCP tools. `save_image_memory`, `query_images`, and `get_image` should be absent.
- **memini-ai:** Check debug logs for `query_memories` — no "image RRF arm" log line appears.
- **Process memory:** No `clip-ViT` model in process memory (`pip list | grep memini-vision` shows nothing if the extra is not installed).

## 7. Configuration

### 7.1 All Environment Variables

| Variable | Server | Default | Description |
|----------|--------|---------|-------------|
| `MEMINI_IMAGE_SEARCH_ENABLED` | Both | `false` | Master gate for the entire image search subsystem. |
| `MEMINI_IMAGE_CLIP_MODEL` | Both | `clip-ViT-B-32` | CLIP model ID. `clip-ViT-B-32` (512-dim) or `clip-ViT-L-14` (768-dim). |
| `MEMINI_IMAGE_CLIP_DEVICE` | Both | `auto` | Device: `auto`, `cpu`, or `cuda`. |
| `MEMINI_IMAGE_DIR` | Both | `~/.memini-ai/images` | Filesystem directory for stored images. |
| `MEMINI_IMAGE_MAX_FILE_SIZE` | Both | `10485760` (10MB) | Max image file size in bytes. |
| `MEMINI_IMAGE_DB_URL` | Both | (empty) | PostgreSQL connection. Falls back to `MEMINI_DB_URL` if empty. |

All variables are consumed via `memini-vision.VisionConfig` (pydantic-settings with `env_prefix="MEMINI_IMAGE_"`).

### 7.2 opencode.json Configuration

To enable image search, add to the `videre-mcp` and `memini-ai-dev` MCP server env blocks:

```json
{
  "MEMINI_IMAGE_SEARCH_ENABLED": "true",
  "MEMINI_IMAGE_CLIP_MODEL": "clip-ViT-B-32",
  "MEMINI_IMAGE_CLIP_DEVICE": "auto"
}
```

When the block is absent or `MEMINI_IMAGE_SEARCH_ENABLED` is `false`, the subsystem is completely dormant.

## 8. Migration, Rollout, Risk

### 8.1 Release Order

Three repos, bumped in dependency order:

| # | Repo | Version | Action |
|---|------|---------|--------|
| 1 | `memini-vision` | **0.1.0** (new) | Create package, publish to PyPI. |
| 2 | `videre-mcp` | 0.2.1 → **0.3.0** | Add `vision` extra, 3 new tools, conditional registration. |
| 3 | `memini-ai-dev` | 0.7.9 → **0.8.0** | Add `vision` extra, schema migration, RRF third arm. |

### 8.2 Schema Migration

- **Additive, idempotent.** `CREATE TABLE IF NOT EXISTS` — safe to run multiple times.
- **~826 existing memories untouched.** The `memories_image` table starts empty. No backfill needed.
- **`source_type` CHECK extension** is a non-disruptive `ALTER TABLE` — existing rows already satisfy the new constraint (it's a superset).
- **Rollback:** `DROP TABLE IF EXISTS memories_image CASCADE;` — zero impact on text memories.

### 8.3 Backwards Compatibility

- **Text-only users (the default) see zero behavior change.** `MEMINI_IMAGE_SEARCH_ENABLED` defaults to `false`.
- `query_memories` API is unchanged. Image search is transparently included when enabled.
- All existing MCP tools continue to work. The 3 new image tools only appear when enabled.
- The `vision` extra is optional — users who don't install `memini-vision` see no change.

### 8.4 Risk Surface

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CLIP model download fails (network) | Low | `ClipEmbedder` raises on first use. videre-mcp tools return error dict. memini-ai logs warning and skips image RRF arm. |
| CLIP RAM usage (~400MB B/32, ~1.5GB L/14) | Low (16GB workstation) | Lazy loading — model only loads when feature is enabled. Default is off. |
| Dim mismatch: ViT-B/32 (512) vs ViT-L/14 (768) | None | Schema uses `vector(768)`. B/32 vectors zero-padded at write time. |
| pgvector compatibility with `vector(768)` | None | pgvector supports dims 1–2000 since v0.5.0. Running container uses pgvector 0.8.x. |
| Two processes loading CLIP (videre-mcp + memini-ai) | Medium | Each process loads its own CLIP instance. Combined RAM: ~800MB (B/32) or ~3GB (L/14). Acceptable for a 16GB workstation. Users on constrained machines should use B/32 or keep the feature off. |
| Model loading latency on first request | Medium | B/32: ~2-5s, L/14: ~5-10s. Subsequent calls use cached model. Acceptable for interactive use. |

### 8.5 Estimated Lines of Code

| Repo | Component | Est. LoC |
|------|-----------|----------|
| **memini-vision** | `config.py`, `store.py`, `embedder.py`, `index.py`, `query.py`, `types.py`, tests | ~800 |
| **videre-mcp** | 3 new tools in `server.py`, conditional registration, `pyproject.toml` | ~200 |
| **memini-ai-dev** | Migration SQL, `source_type` CHECK, RRF third arm in `system.py`, `pyproject.toml` | ~400 |
| **Total** | | **~1,400** |

Test count estimate: ~50 new tests (25 memini-vision unit tests, 15 videre-mcp integration tests, 10 memini-ai RRF tests).

## 9. Decisions Log

All 8 user decisions from the 2026-07-13 design thread. No open questions remain.

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Storage: Filesystem pointer.** Path + sha256 in DB, bytes on disk at `~/.memini-ai/images/`. Sharded by first 2 hex chars of SHA-256. | Matches existing `source_path` pattern in `memories` table. Avoids TOAST bloat, keeps `pg_dump` lean. |
| 2 | **CLIP: ViT-B/32 default, ViT-L/14 opt-in.** Runtime-selectable via `MEMINI_IMAGE_CLIP_MODEL`. Schema uses `vector(768)` to accommodate both. | B/32 is cheaper/faster for default use. L/14 for high-precision recall. Zero-padding avoids separate dim config. |
| 3 | **GPU: `MEMINI_IMAGE_CLIP_DEVICE=auto|cpu|cuda`, default `auto`.** | Mirrors existing `device` field pattern in memini-ai `config.py:31`. |
| 4 | **Auto-screenshot: EXPLICIT ONLY for v0.8.0.** Agent calls `videre-mcp_take_screenshot` then `videre-mcp_save_image_memory`. Auto-screenshot deferred. | Cross-MCP-server orchestration doesn't exist in boomerang-v3 today. Out of scope. |
| 5 | **Retrieval: PATH ONLY.** All tools return absolute filesystem paths. No base64. No `format` option. | Simpler, faster, no encoding overhead. Agent reads the file at the returned path. |
| 6 | **Shared-package architecture.** `memini-vision` is a standalone library (not an MCP server). videre-mcp and memini-ai both depend on it. | Single source of truth for storage + embedding logic. No duplication. videre-mcp is the front door, memini-ai is the back end. |
| 7 | **Three-repo split.** memini-vision (new, 0.1.0) → videre-mcp (0.2.1 → 0.3.0) → memini-ai (0.7.9 → 0.8.0). | Dependency order: shared library first, then consumers. Each repo versioned independently. |
| 8 | **Version bumps.** memini-vision 0.1.0 (new), videre-mcp 0.3.0 (minor: new feature), memini-ai 0.8.0 (minor: additive schema + feature). | All backwards compatible. Text-only users see zero change. |

**Status: READY FOR IMPLEMENTATION.** Zero open questions. All design decisions are binding.
