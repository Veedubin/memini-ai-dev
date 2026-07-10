-- Multi-model embedding support (v0.12.0)
-- Adds:
--   - embedding_model column on memories (text, tracks which model produced embedding)
--   - embedding_bge_large column on memories (1024-dim, parallel to embedding_bge_m3)
--   - index on embedding_bge_large
--
-- Idempotent: safe to run multiple times (IF NOT EXISTS / IF EXISTS on all statements).

-- 1. Add embedding_model column to memories (text, nullable for backwards compat)
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100);

-- 2. Backfill: existing 384-dim memories are assumed to be MiniLM
-- (memini-ai has only ever used MiniLM at 384-dim by default)
UPDATE memories SET embedding_model = 'all-MiniLM-L6-v2'
WHERE embedding IS NOT NULL AND embedding_model IS NULL;

-- 3. Add 1024-dim BGE-Large column (parallel to embedding_bge_m3 in memories_1024)
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_bge_large vector(1024);

-- 4. Add index for BGE-Large (DiskANN preferred; HNSW fallback handled at app level)
CREATE INDEX IF NOT EXISTS idx_memories_embedding_bge_large ON memories
USING diskann (embedding_bge_large vector_cosine_ops);

-- 5. Add index for embedding_model (fast "what models are in use" queries)
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model ON memories (embedding_model)
WHERE embedding_model IS NOT NULL;