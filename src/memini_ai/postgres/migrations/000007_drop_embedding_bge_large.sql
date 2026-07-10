-- Drop embedding_bge_large column (v0.7.6)
--
-- BGE-Large support was added in v0.7.0 but turned out not to be needed in
-- production. v0.7.6 removes it to keep the schema clean.
--
-- The BGE-Large migration script is kept as a reference example for users
-- who want to do similar migrations on their own:
--   archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py
--
-- Idempotent: safe to run multiple times (IF EXISTS on all statements).

-- 1. Drop the BGE-Large index (DiskANN or HNSW)
DROP INDEX IF EXISTS idx_memories_embedding_bge_large;

-- 2. Drop the column itself
ALTER TABLE memories DROP COLUMN IF EXISTS embedding_bge_large;
