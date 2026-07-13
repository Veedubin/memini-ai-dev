-- Add memories_image table for CLIP-based image search (v0.8.0)
--
-- Shared with videre-mcp via the memini-vision library. The schema is
-- owned by memini-ai for backwards compatibility; memini-vision's
-- ImageIndex.ensure_schema() is idempotent and safe to call from
-- either process — whichever starts first creates the table, the
-- other's call is a no-op.
--
-- The table is created at memini-ai startup REGARDLESS of whether
-- MEMINI_IMAGE_SEARCH_ENABLED is true. This ensures videre-mcp can
-- write image rows without memini-ai needing image search enabled.
--
-- Idempotent: safe to run multiple times (IF NOT EXISTS / IF EXISTS
-- on all statements). Re-running produces no errors.

-- 1. Create the memories_image table (1:1 FK to memories, ON DELETE CASCADE)
CREATE TABLE IF NOT EXISTS memories_image (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'placeholder-768',

    image_path TEXT NOT NULL,
    image_sha256 VARCHAR(64) NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    width INT,
    height INT,
    caption TEXT,
    file_size_bytes BIGINT,

    trust_score FLOAT DEFAULT 0.5 CHECK (trust_score >= 0 AND trust_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Vector index (DiskANN preferred; HNSW fallback if vectorscale unavailable).
--    The HNSW variant is created unconditionally here so the table is queryable
--    even without vectorscale; the application's _ensure_schema() will create
--    the DiskANN index instead when vectorscale is detected. Both use the
--    same index name (IF NOT EXISTS makes them mutually exclusive).
CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 3. Secondary indexes
CREATE INDEX IF NOT EXISTS idx_memories_image_memory_id ON memories_image(memory_id);
CREATE INDEX IF NOT EXISTS idx_memories_image_sha256 ON memories_image(image_sha256);
CREATE INDEX IF NOT EXISTS idx_memories_image_trust ON memories_image(trust_score);
CREATE INDEX IF NOT EXISTS idx_memories_image_created_at ON memories_image(created_at DESC);

-- 4. Extend memories.source_type CHECK to include 'image'
--    (superset of the existing constraint — existing rows still satisfy it)
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought', 'image'));