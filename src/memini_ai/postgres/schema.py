"""PostgreSQL schema definitions for pgvector/pgvectorscale backend.

This module defines the full SQL schema for memini-ai's PostgreSQL backend using
pgvector for vector storage and pgvectorscale's StreamingDiskANN index for
high-performance similarity search.

Schema Design Decisions:
- Use vector(1024) for BGE-Large embeddings (384 for MiniLM fallback)
- Use StreamingDiskANN (diskann) index for vector similarity - better for large datasets
- Use vector_cosine_ops for cosine distance similarity
- Enable both pgvector and vectorscale extensions
"""

# Table name constants
TABLE_MEMORIES = "memories"
TABLE_MEMORY_RELATIONSHIPS = "memory_relationships"
TABLE_ENTITIES = "entities"
TABLE_ENTITY_RELATIONSHIPS = "entity_relationships"
TABLE_PEERS = "peers"
TABLE_MEMORY_SHARING = "memory_sharing"
TABLE_USER_PROFILES = "user_profiles"
TABLE_TRUST_ADJUSTMENTS = "trust_adjustments"
TABLE_THOUGHT_CHAINS = "thought_chains"
TABLE_THOUGHTS = "thoughts"

# SQL for creating all extensions
SQL_CREATE_EXTENSIONS = """
-- Enable pgvector extension for vector data type
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable vectorscale extension for StreamingDiskANN index
CREATE EXTENSION IF NOT EXISTS vectorscale;
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
    metadata JSONB DEFAULT '{}'::jsonb
);
"""

# SQL for memories vector index (StreamingDiskANN)
SQL_CREATE_MEMORIES_EMBEDDING_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
USING diskann (embedding vector_cosine_ops);
"""

# SQL for memories secondary indexes
SQL_CREATE_MEMORIES_INDEXES = """
-- Index for trust engine queries
CREATE INDEX IF NOT EXISTS idx_memories_trust ON memories(trust_score) WHERE NOT is_archived;

-- Index for last accessed queries
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed_at) WHERE NOT is_archived;

-- Index for peer queries
CREATE INDEX IF NOT EXISTS idx_memories_peer ON memories(peer_id) WHERE peer_id IS NOT NULL;
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

# SQL for entities vector index (StreamingDiskANN)
SQL_CREATE_ENTITIES_EMBEDDING_INDEX = """
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
USING diskann (embedding vector_cosine_ops);
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
CREATE INDEX IF NOT EXISTS idx_thoughts_embedding ON thoughts USING diskann (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_thoughts_branch ON thoughts(branch_id) WHERE branch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thoughts_revises ON thoughts(revises_thought_id) WHERE revises_thought_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_thoughts_memory ON thoughts(memory_id) WHERE memory_id IS NOT NULL;
"""

# SQL to update memories source_type CHECK constraint to include 'thought'
SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK = """
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought'));
"""


def get_schema_sql() -> str:
    """Return all SQL schema definitions as a single concatenated string.

    Returns:
        Complete SQL script for creating all tables, indexes, and extensions
        for the pgvector/pgvectorscale backend.
    """
    return "\n".join(
        [
            SQL_CREATE_EXTENSIONS,
            SQL_CREATE_PEERS_TABLE,  # Must be first - other tables reference it
            SQL_CREATE_MEMORIES_TABLE,
            SQL_CREATE_MEMORIES_EMBEDDING_INDEX,
            SQL_CREATE_MEMORIES_INDEXES,
            SQL_CREATE_MEMORY_RELATIONSHIPS_TABLE,
            SQL_CREATE_MEMORY_RELATIONSHIPS_INDEXES,
            SQL_CREATE_ENTITIES_TABLE,  # References peers
            SQL_CREATE_ENTITIES_EMBEDDING_INDEX,
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
            SQL_CREATE_THOUGHTS_INDEXES,
            # Update memories source_type CHECK constraint
            SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK,
        ]
    )
