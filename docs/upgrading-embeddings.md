# Upgrading Embeddings: MiniLM → BGE-M3

This guide covers why and when to upgrade from MiniLM-L6-v2 (384-dim) to
BGE-M3 (1024-dim), a 4-step migration recipe, rollback instructions,
new-deployment guidance, and an FAQ.

## Why Upgrade?

| Aspect | MiniLM-L6-v2 | BGE-M3 |
|--------|-------------|--------|
| Dimensions | 384 | 1024 |
| Model size | ~80 MB | ~2.3 GB |
| CPU speed | Fast | Slower |
| GPU benefit | Minimal | Significant |
| Precision | Good | Better (higher recall on complex queries) |

BGE-M3 produces richer semantic vectors, improving recall on long,
complex, or multi-sentence queries. The trade-off is model size and
inference speed — BGE-M3 is best on a GPU.

## When to Upgrade

- You have a GPU available (or are on a machine with sufficient RAM for
  CPU inference at acceptable latency).
- Your dataset is large enough that the precision difference matters
  (>1,000 memories).
- You're starting a **new deployment** (v0.7.7+ auto-detects BGE-M3 on
  empty databases — no action needed).

## 4-Step Migration Recipe

> **⚠️ Order matters.** Don't change your env vars BEFORE the migration
> script runs — that creates a dim mismatch (`BGE-M3` produces 1024-dim
> vectors but the `embedding` column is still 384-dim) and breaks the
> server. v0.7.7 downgrades this to a warning + text-only fallback, but
> you should still avoid the mismatch. **Migrate first, then change env vars.**

### Step 1: Backup your database

```bash
pg_dump -h localhost -p 5434 -U postgres -d postgres \
  -Fc -f memini-backup-$(date +%Y%m%d).dump
```

### Step 2: Ensure torch with CUDA support is installed (sentence-transformers[gpu] is not a valid pip extra)

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (default), torch is already a core dependency — no action needed.

### Step 3: Run the migration script

```bash
python archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py \
  --db-url "postgresql://postgres:password@localhost:5434/postgres"
```

The script is **non-destructive**: your original 384-dim vectors are
preserved in the `embedding` column. The script generates new 1024-dim
vectors and writes them to the `embedding_bge_m3` column. This step may
take several minutes for large databases (~1,000 memories/minute on CPU,
much faster on GPU).

### Step 4: Update env vars, restart, and verify

```bash
# Add to your .env or opencode.json memini-ai-dev.environment
export MEMINI_MODEL_NAME=BAAI/bge-m3
export MEMINI_EMBEDDING_DIM=1024
# Optional: enable RRF to search across both MiniLM and BGE-M3 columns
export MEMINI_ENABLE_RRF=true
```

Then restart memini-ai. After restart, call `get_status` and confirm:
- `modelName: "BAAI/bge-m3"`
- `modelDimension: 1024`
- `embeddingDimMismatch: false`

If `embeddingDimMismatch` is `true`, the model loaded but the dim is
still wrong — re-check your env vars and the migration script.

## Rollback

To revert to MiniLM:

1. Unset `MEMINI_MODEL_NAME` (or set it to `all-MiniLM-L6-v2`).
2. Set `MEMINI_EMBEDDING_DIM=384`.
3. Restart the server.

The original 384-dim vectors in the `embedding` column are still intact.
The 1024-dim vectors in `embedding_bge_m3` remain in the DB but are
unused. You can drop the `embedding_bge_m3` column if desired, but it's
not necessary.

## New Deployments

Starting with v0.7.7, memini-ai auto-detects new deployments (0 memories
in the database) and defaults to BGE-M3. To keep MiniLM instead:

```bash
export MEMINI_AUTO_DETECT_MODEL=false
# or
export MEMINI_MODEL_NAME=all-MiniLM-L6-v2
```

## FAQ

**Q: Will upgrading delete my existing memories?**

No. The migration script is non-destructive. Your 384-dim vectors stay
in the `embedding` column. The script adds 1024-dim vectors to the
`embedding_bge_m3` column.

**Q: Can I use both models simultaneously?**

Yes — with `EMBEDDING_MODE=auto`, memini-ai uses Reciprocal Rank Fusion
(RRF) to combine search results from both vector spaces. See the
dual-model RRF documentation for details.

**Q: What if my model and embedding dim don't match?**

Since v0.7.7, a dim mismatch no longer crashes the server. Instead, the
server logs a warning and degrades to text-only search. Vector search is
disabled until you fix the mismatch (set `MEMINI_EMBEDDING_DIM` to match
your model's output dim). Set `MEMINI_STRICT_EMBEDDING_DIM=true` to
restore the old crash-on-mismatch behavior.

**Q: How do I check if there's a mismatch?**

Call `get_status` — it now reports `embeddingDimMismatch`,
`embeddingDimExpected`, `embeddingDimActual`, `modelName`, and
`modelDimension`.