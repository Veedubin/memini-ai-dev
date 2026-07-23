"""PostgreSQL schema definitions for pgvector/pgvectorscale backend.

This module defines the full SQL schema for memini-ai's PostgreSQL backend using
pgvector for vector storage and pgvectorscale's StreamingDiskANN index for
high-performance similarity search.

Schema Design Decisions:
- Use vector(384) for MiniLM-L6-v2 embeddings (default), vector(1024) for BGE-M3
- Use StreamingDiskANN (diskann) index when vectorscale is available, fall back to HNSW
- Use vector_cosine_ops for cosine distance similarity
- Enable pgvector (required) and vectorscale (optional) extensions
"""

# Table name constants
TABLE_MEMORIES = "memories"
TABLE_MEMORIES_1024 = "memories_1024"
TABLE_MEMORIES_IMAGE = "memories_image"
TABLE_MEMORY_RELATIONSHIPS = "memory_relationships"
TABLE_ENTITIES = "entities"
TABLE_ENTITY_RELATIONSHIPS = "entity_relationships"
TABLE_PEERS = "peers"
TABLE_MEMORY_SHARING = "memory_sharing"
TABLE_USER_PROFILES = "user_profiles"
TABLE_TRUST_ADJUSTMENTS = "trust_adjustments"
TABLE_THOUGHT_CHAINS = "thought_chains"
TABLE_THOUGHTS = "thoughts"
TABLE_AUDIT_LOG = "audit_log"
TABLE_KANBAN_CARDS = "kanban_cards"

# SQL for creating all extensions
# pgvector is required; vectorscale is optional (fall back to HNSW if unavailable)
SQL_CREATE_EXTENSIONS = """
-- Enable pgvector extension for vector data type (required)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable vectorscale extension for StreamingDiskANN index (optional)
-- Fail gracefully if unavailable — we'll use HNSW index instead.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vectorscale;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'vectorscale extension unavailable, will use HNSW indexes';
END $$;
"""

# SQL for memories table
SQL_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    embedding vector(384),

    -- Source tracking
    source_type VARCHAR(50) NOT NULL CHECK (
        source_type IN ('session', 'file', 'web', 'boomerang', 'project')
    ),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Trust Engine fields
    trust_score FLOAT DEFAULT 0.5 CHECK (trust_score >= 0 AND trust_score <= 1),
    retrieval_count INT DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE,
    last_accessed_at TIMESTAMP WITH TIME ZONE,

    -- Multi-peer support
    peer_id UUID REFERENCES peers(id) ON DELETE SET NULL,

    -- Content deduplication
    content_hash VARCHAR(64),
    source_path TEXT,

    -- Flexible metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Delta model fields (v0.4.0)
    created_at_ms BIGINT DEFAULT 0,
    supersedes_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    structured_fields JSONB DEFAULT NULL,
    change_ratio FLOAT DEFAULT 1.0 CHECK (change_ratio >= 0 AND change_ratio <= 1),

    -- Multi-model embedding support (v0.12.0+)
    -- Tracks which model produced the primary embedding column.
    -- Nullable for backwards compat with pre-v0.12.0 rows.
    embedding_model VARCHAR(100)
);
"""

# SQL for memories vector index (StreamingDiskANN preferred, HNSW fallback)
SQL_CREATE_MEMORIES_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
USING diskann (embedding vector_cosine_ops);
"""

SQL_CREATE_MEMORIES_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# SQL for memories secondary indexes
SQL_CREATE_MEMORIES_INDEXES = """
-- Index for trust engine queries
CREATE INDEX IF NOT EXISTS idx_memories_trust ON memories(trust_score) WHERE NOT is_archived;

-- Index for last accessed queries
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at) WHERE NOT is_archived;

-- Index for peer queries
CREATE INDEX IF NOT EXISTS idx_memories_peer ON memories(peer_id) WHERE peer_id IS NOT NULL;

-- Multi-model (v0.12.0+): embedding_model index for "what models are in use" queries
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model ON memories (embedding_model)
WHERE embedding_model IS NOT NULL;
"""

# =============================================================================
# memories_1024 table (v0.7.0 Dual-Model RRF)
# =============================================================================
#
# The memories_1024 table holds high-dimensional embeddings (1024-dim)
# for memories that have been "elevated" from the default 384-dim MiniLM space.
# Each row is FK-linked to the corresponding row in the memories table, so the
# 384-dim record is always the source of truth and the 1024-dim record is a
# quality-boost sidecar. Idempotent migration: existing 384-dim memories are
# NOT touched. This table is empty until the elevate_memory_to_1024 tool is used.

# SQL for memories_1024 table
SQL_CREATE_MEMORIES_1024_TABLE = """
CREATE TABLE IF NOT EXISTS memories_1024 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign key to the canonical 384-dim record (source of truth).
    -- ON DELETE CASCADE: if the source memory is hard-deleted, the 1024 copy goes with it.
    -- The 384-dim record itself is soft-deleted (is_archived=TRUE) in normal use.
    memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,

    -- 1024-dim embedding vector
    embedding vector(1024) NOT NULL,

    -- Elevation metadata
    elevated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    elevated_from_dim INT DEFAULT 384,
    embedding_model VARCHAR(100) DEFAULT 'placeholder-1024',

    -- Mirrored trust boost: this 1024 copy inherits and may extend the trust score.
    -- The 384-dim record is the canonical trust source; this is a denormalized cache.
    trust_score FLOAT DEFAULT 0.5 CHECK (trust_score >= 0 AND trust_score <= 1)
);
"""

# SQL for memories_1024 vector index (StreamingDiskANN preferred, HNSW fallback)
SQL_CREATE_MEMORIES_1024_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_memories_1024_embedding ON memories_1024
USING diskann (embedding vector_cosine_ops);
"""

SQL_CREATE_MEMORIES_1024_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_memories_1024_embedding ON memories_1024
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# SQL for memories_1024 secondary indexes
SQL_CREATE_MEMORIES_1024_INDEXES = """
-- Index for joining back to memories (most queries will join on memory_id)
CREATE INDEX IF NOT EXISTS idx_memories_1024_memory_id ON memories_1024(memory_id);

-- Index for trust-based filtering
CREATE INDEX IF NOT EXISTS idx_memories_1024_trust ON memories_1024(trust_score);

-- Index for elevation timestamp (for "recently elevated" queries)
CREATE INDEX IF NOT EXISTS idx_memories_1024_elevated_at ON memories_1024(elevated_at DESC);
"""

# =============================================================================
# memories_image table (v0.8.0 Image Recall RRF)
# =============================================================================
#
# The memories_image table holds CLIP image embeddings (768-dim, accommodating
# both ViT-B/32 zero-padded to 768 and ViT-L/14 native 768) for memories that
# have an associated image (screenshots, diagrams, etc.). Each row is
# 1:1 FK-linked to the corresponding row in the memories table, so the
# text record is always the source of truth and the image row is a
# cross-modal recall sidecar. The table is created at memini-ai startup
# REGARDLESS of whether MEMINI_IMAGE_SEARCH_ENABLED is true — this lets
# videre-mcp write image rows without memini-ai needing image search on.
# Idempotent migration: existing memories are NOT touched. This table is
# empty until the videre-mcp save_image_memory tool is used.

# SQL for memories_image table
SQL_CREATE_MEMORIES_IMAGE_TABLE = """
CREATE TABLE IF NOT EXISTS memories_image (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign key to the canonical memories record (source of truth).
    -- ON DELETE CASCADE: if the source memory is hard-deleted, the image
    -- row goes with it. 1:1 (UNIQUE) — one image per memory.
    memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,

    -- 768-dim CLIP embedding vector (accommodates both ViT-B/32 zero-padded
    -- to 768 and ViT-L/14 native 768).
    embedding vector(768) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'placeholder-768',

    -- Image metadata (filesystem-pointer storage — bytes on disk, path in DB)
    image_path TEXT NOT NULL,
    image_sha256 VARCHAR(64) NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    width INT,
    height INT,
    caption TEXT,
    file_size_bytes BIGINT,

    -- Mirrored trust score (denormalized cache; memories.trust_score is canonical)
    trust_score FLOAT DEFAULT 0.5 CHECK (trust_score >= 0 AND trust_score <= 1),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for memories_image vector index (StreamingDiskANN preferred, HNSW fallback)
SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
USING diskann (embedding vector_cosine_ops);
"""

SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_memories_image_embedding ON memories_image
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# SQL for memories_image secondary indexes
SQL_CREATE_MEMORIES_IMAGE_INDEXES = """
-- Index for joining back to memories (most queries will join on memory_id)
CREATE INDEX IF NOT EXISTS idx_memories_image_memory_id ON memories_image(memory_id);

-- Index for sha256-based idempotent re-insertion checks
CREATE INDEX IF NOT EXISTS idx_memories_image_sha256 ON memories_image(image_sha256);

-- Index for trust-based filtering
CREATE INDEX IF NOT EXISTS idx_memories_image_trust ON memories_image(trust_score);

-- Index for creation timestamp (for "recently added image" queries)
CREATE INDEX IF NOT EXISTS idx_memories_image_created_at ON memories_image(created_at DESC);
"""

# SQL to extend memories source_type CHECK constraint to include 'image' + 'github'
SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE = """
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought', 'image', 'github'));
"""

# SQL for memory_relationships table
SQL_CREATE_MEMORY_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS memory_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL CHECK (
        relationship_type IN ('SUPERSEDES', 'RELATED_TO', 'CONTRADICTS', 'DERIVED_FROM')
    ),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Prevent duplicate relationships
    UNIQUE(source_id, target_id, relationship_type)
);
"""

# SQL for memory_relationships indexes
SQL_CREATE_MEMORY_RELATIONSHIPS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_mem_rel_source ON memory_relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_mem_rel_target ON memory_relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_mem_rel_type ON memory_relationships(relationship_type);
"""

# SQL for entities table
SQL_CREATE_ENTITIES_TABLE = """
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(50) NOT NULL CHECK (
        entity_type IN ('PERSON', 'ORGANIZATION', 'CONCEPT', 'CODE', 'PROJECT', 'LOCATION', 'UNKNOWN')
    ),
    canonical_name VARCHAR(500),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Vector embedding for entity similarity
    embedding vector(384),

    -- Ownership
    peer_id UUID REFERENCES peers(id) ON DELETE SET NULL,

    -- Occurrence tracking
    mention_count INT DEFAULT 1,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(name, entity_type, peer_id)
);
"""

# SQL for entities vector index (StreamingDiskANN preferred, HNSW fallback)
SQL_CREATE_ENTITIES_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
USING diskann (embedding vector_cosine_ops);
"""

SQL_CREATE_ENTITIES_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# SQL for entities indexes
SQL_CREATE_ENTITIES_INDEXES = """
-- Index for peer queries
CREATE INDEX IF NOT EXISTS idx_entities_peer ON entities(peer_id) WHERE peer_id IS NOT NULL;

-- Index for entity type queries
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
"""

# SQL for entity_relationships table
SQL_CREATE_ENTITY_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS entity_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);
"""

# SQL for entity_relationships indexes
SQL_CREATE_ENTITY_RELATIONSHIPS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ent_rel_source ON entity_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_ent_rel_target ON entity_relationships(target_entity_id);
"""

# SQL for peers table
SQL_CREATE_PEERS_TABLE = """
CREATE TABLE IF NOT EXISTS peers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'COLLABORATOR' CHECK (
        role IN ('OWNER', 'COLLABORATOR', 'READONLY', 'GUEST')
    ),
    trust_level FLOAT DEFAULT 1.0 CHECK (trust_level >= 0 AND trust_level <= 1),
    preferences JSONB DEFAULT '{}'::jsonb,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE
);
"""

# SQL for memory_sharing table
SQL_CREATE_MEMORY_SHARING_TABLE = """
CREATE TABLE IF NOT EXISTS memory_sharing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    peer_id UUID NOT NULL REFERENCES peers(id) ON DELETE CASCADE,
    permission VARCHAR(50) NOT NULL DEFAULT 'SHARED' CHECK (
        permission IN ('PRIVATE', 'SHARED', 'INHERITED')
    ),
    granted_by UUID REFERENCES peers(id) ON DELETE SET NULL,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(memory_id, peer_id)
);
"""

# SQL for memory_sharing indexes
SQL_CREATE_MEMORY_SHARING_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_mem_sharing_memory ON memory_sharing(memory_id);
CREATE INDEX IF NOT EXISTS idx_mem_sharing_peer ON memory_sharing(peer_id);
"""

# SQL for user_profiles table
SQL_CREATE_USER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id UUID UNIQUE REFERENCES peers(id) ON DELETE CASCADE,

    -- Profile data
    preferences JSONB DEFAULT '{}'::jsonb,
    communication_style VARCHAR(100) DEFAULT 'neutral',
    expertise_level VARCHAR(50) DEFAULT 'intermediate',

    -- Dialectic notes for LLM reasoning traces
    dialectic_notes JSONB DEFAULT '[]'::jsonb,

    -- Status
    warmed_up BOOLEAN DEFAULT FALSE,
    session_count INT DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for user_profiles indexes
SQL_CREATE_USER_PROFILES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_user_profiles_peer ON user_profiles(peer_id) WHERE peer_id IS NOT NULL;
"""

# SQL for trust_adjustments table
SQL_CREATE_TRUST_ADJUSTMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS trust_adjustments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    old_score FLOAT NOT NULL,
    new_score FLOAT NOT NULL,
    signal VARCHAR(50) NOT NULL CHECK (
        signal IN ('agent_used', 'agent_ignored', 'user_corrected', 'user_confirmed')
    ),
    adjustment_amount FLOAT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for trust_adjustments indexes
SQL_CREATE_TRUST_ADJUSTMENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_trust_adj_memory ON trust_adjustments(memory_id);
CREATE INDEX IF NOT EXISTS idx_trust_adj_created ON trust_adjustments(created_at);
"""

# =============================================================================
# Thought Chains tables (Phase 5)
# =============================================================================

# SQL for thought_chains table
SQL_CREATE_THOUGHT_CHAINS_TABLE = """
CREATE TABLE IF NOT EXISTS thought_chains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    parent_chain_id UUID REFERENCES thought_chains(id) ON DELETE SET NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'abandoned')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for thought_chains indexes
SQL_CREATE_THOUGHT_CHAINS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_thought_chains_session ON thought_chains(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thought_chains_parent ON thought_chains(parent_chain_id) WHERE parent_chain_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thought_chains_status ON thought_chains(status) WHERE status = 'active';
"""

# SQL for thoughts table
SQL_CREATE_THOUGHTS_TABLE = """
CREATE TABLE IF NOT EXISTS thoughts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id UUID NOT NULL REFERENCES thought_chains(id) ON DELETE CASCADE,

    -- Original sequential-thinking fields
    thought TEXT NOT NULL,
    thought_number INTEGER NOT NULL CHECK (thought_number >= 1),
    total_thoughts INTEGER NOT NULL CHECK (total_thoughts >= 1),
    next_thought_needed BOOLEAN NOT NULL,

    -- Revision support
    is_revision BOOLEAN DEFAULT FALSE,
    revises_thought_id UUID REFERENCES thoughts(id) ON DELETE SET NULL,

    -- Branching support
    branch_from_thought_id UUID REFERENCES thoughts(id) ON DELETE SET NULL,
    branch_id VARCHAR(255),

    -- memini-ai additions
    embedding vector(384),
    content_hash VARCHAR(64),
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for thoughts indexes
SQL_CREATE_THOUGHTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_thoughts_chain ON thoughts(chain_id);
CREATE INDEX IF NOT EXISTS idx_thoughts_branch ON thoughts(branch_id) WHERE branch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thoughts_revises ON thoughts(revises_thought_id) WHERE revises_thought_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thoughts_memory ON thoughts(memory_id) WHERE memory_id IS NOT NULL;
"""

SQL_CREATE_THOUGHTS_EMBEDDING_INDEX_DISKANN = """
CREATE INDEX IF NOT EXISTS idx_thoughts_embedding ON thoughts USING diskann (embedding vector_cosine_ops);
"""

SQL_CREATE_THOUGHTS_EMBEDDING_INDEX_HNSW = """
CREATE INDEX IF NOT EXISTS idx_thoughts_embedding ON thoughts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# =============================================================================
# Audit Log table (Phase 2.3: Security Audit Logging)
# =============================================================================

# SQL for audit_log table
SQL_CREATE_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'auth_failure', 'permission_change', 'config_modification',
        'agent_execution', 'memory_mutation', 'tool_invocation', 'trust_adjustment'
    )),
    severity VARCHAR(20) NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'critical')),
    session_id UUID,
    peer_id VARCHAR(100),
    agent_name VARCHAR(100),
    tool_name VARCHAR(100),
    memory_id UUID,
    description TEXT,
    details JSONB,
    state_before JSONB,
    state_after JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

# SQL for audit_log indexes
SQL_CREATE_AUDIT_LOG_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_audit_log_occurred_at ON audit_log(occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at_brin ON audit_log USING BRIN(created_at);
"""

# SQL to update memories source_type CHECK constraint to include 'thought' + 'image' + 'github'
SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK = """
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought', 'image', 'github'));
"""

# =============================================================================
# kanban_cards table (GitHub triage poller integration)
# =============================================================================
#
# Plain Postgres rows — NO pgvector column. Cards are structured data
# (issue/PR metadata + wrapped prompt text), not embeddings. The wrapped
# issue/PR text is separately embedded as a memory (source_type='github')
# via add_memory; the optional memory_id FK links the card to that
# embedded memory (ticket ↔ issue ↔ memory linkage).
#
# Created at memini-ai startup (idempotent IF NOT EXISTS). The GitHub
# triage poller (scripts/gh-triage-poller.py) inserts cards here on each
# poll. ON CONFLICT (repo, number, item_type) DO NOTHING makes re-polls
# idempotent.

SQL_CREATE_KANBAN_CARDS_TABLE = """
CREATE TABLE IF NOT EXISTS kanban_cards (
    card_id      TEXT PRIMARY KEY,
    repo         TEXT NOT NULL,
    number       INTEGER NOT NULL,
    item_type    TEXT NOT NULL CHECK (
        item_type IN ('bug', 'feature', 'question', 'docs', 'pr', 'triage')
    ),
    status       TEXT NOT NULL DEFAULT 'triage' CHECK (
        status IN ('triage', 'todo', 'ready', 'running', 'blocked', 'done', 'archived')
    ),
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    author       TEXT,
    wrapped_text TEXT,
    draft        BOOLEAN DEFAULT FALSE,
    memory_id    UUID REFERENCES memories(id) ON DELETE SET NULL,
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (repo, number, item_type)
);
"""

SQL_CREATE_KANBAN_CARDS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kanban_cards_status ON kanban_cards (status);
CREATE INDEX IF NOT EXISTS idx_kanban_cards_repo ON kanban_cards (repo);
CREATE INDEX IF NOT EXISTS idx_kanban_cards_created ON kanban_cards (created_at);
"""


def get_schema_sql(use_vectorscale: bool = True) -> str:
    """Return all SQL schema definitions as a single concatenated string.

    Args:
        use_vectorscale: If True, use StreamingDiskANN indexes (requires vectorscale).
            If False, use HNSW indexes (requires only pgvector). Default True
            for backward compatibility; auto-detected at runtime by database.py.

    Returns:
        Complete SQL script for creating all tables, indexes, and extensions
        for the pgvector/pgvectorscale backend.
    """
    memories_embedding_index = (
        SQL_CREATE_MEMORIES_EMBEDDING_INDEX_DISKANN
        if use_vectorscale
        else SQL_CREATE_MEMORIES_EMBEDDING_INDEX_HNSW
    )
    entities_embedding_index = (
        SQL_CREATE_ENTITIES_EMBEDDING_INDEX_DISKANN
        if use_vectorscale
        else SQL_CREATE_ENTITIES_EMBEDDING_INDEX_HNSW
    )
    thoughts_embedding_index = (
        SQL_CREATE_THOUGHTS_EMBEDDING_INDEX_DISKANN
        if use_vectorscale
        else SQL_CREATE_THOUGHTS_EMBEDDING_INDEX_HNSW
    )
    memories_1024_embedding_index = (
        SQL_CREATE_MEMORIES_1024_EMBEDDING_INDEX_DISKANN
        if use_vectorscale
        else SQL_CREATE_MEMORIES_1024_EMBEDDING_INDEX_HNSW
    )
    memories_image_embedding_index = (
        SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN
        if use_vectorscale
        else SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW
    )

    return "\n".join(
        [
            SQL_CREATE_EXTENSIONS,
            SQL_CREATE_PEERS_TABLE,  # Must be first - other tables reference it
            SQL_CREATE_MEMORIES_TABLE,
            memories_embedding_index,
            SQL_CREATE_MEMORIES_INDEXES,
            # v0.7.0: Dual-model RRF — must come AFTER memories (FK target)
            SQL_CREATE_MEMORIES_1024_TABLE,
            memories_1024_embedding_index,
            SQL_CREATE_MEMORIES_1024_INDEXES,
            # v0.8.0: Image recall RRF — must come AFTER memories (FK target)
            SQL_CREATE_MEMORIES_IMAGE_TABLE,
            memories_image_embedding_index,
            SQL_CREATE_MEMORIES_IMAGE_INDEXES,
            SQL_CREATE_MEMORY_RELATIONSHIPS_TABLE,
            SQL_CREATE_MEMORY_RELATIONSHIPS_INDEXES,
            SQL_CREATE_ENTITIES_TABLE,  # References peers
            entities_embedding_index,
            SQL_CREATE_ENTITIES_INDEXES,
            SQL_CREATE_ENTITY_RELATIONSHIPS_TABLE,
            SQL_CREATE_ENTITY_RELATIONSHIPS_INDEXES,
            SQL_CREATE_MEMORY_SHARING_TABLE,  # References peers
            SQL_CREATE_MEMORY_SHARING_INDEXES,
            SQL_CREATE_USER_PROFILES_TABLE,
            SQL_CREATE_USER_PROFILES_INDEXES,
            SQL_CREATE_TRUST_ADJUSTMENTS_TABLE,
            SQL_CREATE_TRUST_ADJUSTMENTS_INDEXES,
            # Phase 5: Thought Chains
            SQL_CREATE_THOUGHT_CHAINS_TABLE,
            SQL_CREATE_THOUGHT_CHAINS_INDEXES,
            SQL_CREATE_THOUGHTS_TABLE,
            thoughts_embedding_index,
            SQL_CREATE_THOUGHTS_INDEXES,
            # Phase 2.3: Audit Log
            SQL_CREATE_AUDIT_LOG_TABLE,
            SQL_CREATE_AUDIT_LOG_INDEXES,
            # GitHub triage poller: kanban cards (plain rows, FK to memories)
            SQL_CREATE_KANBAN_CARDS_TABLE,
            SQL_CREATE_KANBAN_CARDS_INDEXES,
            # Update memories source_type CHECK constraint (includes 'thought' + 'image' + 'github')
            SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE,
        ]
    )
