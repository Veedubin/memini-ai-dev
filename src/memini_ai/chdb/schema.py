"""chDB schema definitions for memini-ai.

Mirrors ``src/memini_ai/postgres/schema.py`` (the pgvector/pgvectorscale
backend) but targets ClickHouse's MergeTree engine via the chDB Python
package.

Schema design notes
--------------------
- All CREATE TABLE statements are **idempotent** (use IF NOT EXISTS).
- All vector columns are ``Array(Float32)`` (chDB 4.2.1 / ClickHouse 26.5.1.1
  does not include the new ``VECTOR`` type nor the ``vector_similarity``
  HNSW index type. Vector search is brute-force cosine distance.
  Re-evaluate when chDB ships a build with the index type registered.)
- Source-type checks and FK CASCADE are enforced at the application
  layer (chDB has no native CHECK or FK enforcement).
- Dates stored as ``DateTime64(9, 'UTC')`` (microsecond precision, UTC).
  chDB has no real tz support; we store UTC and convert in app code.
- Enums (e.g. source_type, role, permission) use ``LowCardinality(String)``
  for compression. App-layer validates the allowed values.
- Indexes: ``minmax`` for scalar ranges, ``set`` for low-cardinality
  categories, ``bloom_filter`` is available but we don't need it yet.
- ENGINE = MergeTree() ORDER BY id is the standard "small dim" pattern
  for point lookups. For 80 memories this is fine; for 1M we'd consider
  partitioning.

Run order (in ``ChdbDatabase.initialize()``): create all tables, then
add all indexes. Indexes are added separately because ClickHouse can't
ALTER TABLE to add a secondary index inside a CREATE TABLE block when
the index doesn't exist yet on the engine.

See ``docs/memini-ai-v1-chdb-migration.md`` Section 2 for the full
table-by-table mapping from the Postgres schema.
"""

from __future__ import annotations

# =============================================================================
# Table name constants
# =============================================================================
# These match the postgres schema so the public MCP surface can stay
# identical (table_name is referenced by some queries and tools).

TABLE_PEERS = "peers"
TABLE_MEMORIES = "memories"
TABLE_MEMORIES_1024 = "memories_1024"
TABLE_MEMORIES_IMAGE = "memories_image"
TABLE_MEMORY_RELATIONSHIPS = "memory_relationships"
TABLE_ENTITIES = "entities"
TABLE_ENTITY_RELATIONSHIPS = "entity_relationships"
TABLE_MEMORY_SHARING = "memory_sharing"
TABLE_USER_PROFILES = "user_profiles"
TABLE_TRUST_ADJUSTMENTS = "trust_adjustments"
TABLE_THOUGHT_CHAINS = "thought_chains"
TABLE_THOUGHTS = "thoughts"
TABLE_AUDIT_LOG = "audit_log"

ALL_TABLES: list[str] = [
    TABLE_PEERS,
    TABLE_MEMORIES,
    TABLE_MEMORIES_1024,
    TABLE_MEMORIES_IMAGE,
    TABLE_MEMORY_RELATIONSHIPS,
    TABLE_ENTITIES,
    TABLE_ENTITY_RELATIONSHIPS,
    TABLE_MEMORY_SHARING,
    TABLE_USER_PROFILES,
    TABLE_TRUST_ADJUSTMENTS,
    TABLE_THOUGHT_CHAINS,
    TABLE_THOUGHTS,
    TABLE_AUDIT_LOG,
]

# Source-type enum values (validated in app code, not in DDL).
# Mirrors the Postgres CHECK constraint on memories.source_type.
VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    {"session", "file", "web", "boomerang", "project", "thought", "image"}
)

# Valid peer roles.
VALID_PEER_ROLES: frozenset[str] = frozenset(
    {"OWNER", "COLLABORATOR", "READONLY", "GUEST"}
)

# Valid relationship types for memory_relationships.
VALID_MEMORY_RELATIONSHIPS: frozenset[str] = frozenset(
    {"SUPERSEDES", "RELATED_TO", "CONTRADICTS", "DERIVED_FROM"}
)

# Valid entity_type values for entities.
VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {"PERSON", "ORGANIZATION", "CONCEPT", "CODE", "PROJECT", "LOCATION", "UNKNOWN"}
)

# Valid permission values for memory_sharing.
VALID_PERMISSIONS: frozenset[str] = frozenset({"PRIVATE", "SHARED", "INHERITED"})

# Valid trust-adjustment signals.
VALID_TRUST_SIGNALS: frozenset[str] = frozenset(
    {"agent_used", "agent_ignored", "user_corrected", "user_confirmed"}
)

# Valid thought_chain.status values.
VALID_CHAIN_STATUSES: frozenset[str] = frozenset(
    {"active", "paused", "completed", "abandoned"}
)

# Valid audit_log.event_type values.
VALID_AUDIT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "auth_failure",
        "permission_change",
        "config_modification",
        "agent_execution",
        "memory_mutation",
        "tool_invocation",
        "trust_adjustment",
    }
)

# Valid audit_log.severity values.
VALID_AUDIT_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})


# =============================================================================
# CREATE TABLE statements
# =============================================================================
# One per table. All idempotent (IF NOT EXISTS). No FKs (app-layer).
# No CHECK constraints (app-layer). All dates DateTime64(9, 'UTC').

SQL_CREATE_PEERS = """
CREATE TABLE IF NOT EXISTS peers (
    id UUID DEFAULT generateUUIDv4(),
    name String NOT NULL,
    role LowCardinality(String) DEFAULT 'COLLABORATOR',
    trust_level Float64 DEFAULT 1.0,
    preferences JSON DEFAULT '{}',
    is_active Bool DEFAULT true,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    last_active_at Nullable(DateTime64(9, 'UTC')),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id UUID DEFAULT generateUUIDv4(),
    text String NOT NULL,
    -- chDB doesn't allow Nullable(Array(T)). We use an empty array []
    -- as the "no embedding" sentinel. App layer checks length().
    embedding Array(Float32),

    source_type LowCardinality(String) NOT NULL,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    updated_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),

    trust_score Float64 DEFAULT 0.5,
    retrieval_count Int32 DEFAULT 0,
    is_archived Bool DEFAULT false,
    last_accessed_at Nullable(DateTime64(9, 'UTC')),

    peer_id Nullable(UUID),
    content_hash Nullable(String),
    source_path Nullable(String),
    metadata JSON DEFAULT '{}',

    created_at_ms Int64 DEFAULT 0,
    supersedes_id Nullable(UUID),
    structured_fields Nullable(JSON),
    change_ratio Float64 DEFAULT 1.0,
    embedding_model Nullable(String),

    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_MEMORIES_1024 = """
CREATE TABLE IF NOT EXISTS memories_1024 (
    id UUID DEFAULT generateUUIDv4(),
    memory_id UUID NOT NULL,
    embedding Array(Float32) NOT NULL,
    elevated_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    elevated_from_dim Int32 DEFAULT 384,
    embedding_model String DEFAULT 'placeholder-1024',
    trust_score Float64 DEFAULT 0.5,
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_MEMORIES_IMAGE = """
CREATE TABLE IF NOT EXISTS memories_image (
    id UUID DEFAULT generateUUIDv4(),
    memory_id UUID NOT NULL,
    embedding Array(Float32) NOT NULL,
    embedding_model String DEFAULT 'placeholder-768',
    image_path String NOT NULL,
    image_sha256 String NOT NULL,
    mime_type LowCardinality(String) NOT NULL,
    width Nullable(Int32),
    height Nullable(Int32),
    caption Nullable(String),
    file_size_bytes Nullable(Int64),
    trust_score Float64 DEFAULT 0.5,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_MEMORY_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS memory_relationships (
    id UUID DEFAULT generateUUIDv4(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    relationship_type LowCardinality(String) NOT NULL,
    confidence Float64 DEFAULT 1.0,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    metadata JSON DEFAULT '{}',
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id UUID DEFAULT generateUUIDv4(),
    name String NOT NULL,
    entity_type LowCardinality(String) NOT NULL,
    canonical_name Nullable(String),
    confidence Float64 DEFAULT 1.0,
    -- chDB doesn't allow Nullable(Array(T)). Empty array = no embedding.
    embedding Array(Float32),
    peer_id Nullable(UUID),
    mention_count Int32 DEFAULT 1,
    first_seen_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    last_seen_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_ENTITY_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS entity_relationships (
    id UUID DEFAULT generateUUIDv4(),
    source_entity_id UUID NOT NULL,
    target_entity_id UUID NOT NULL,
    relationship_type LowCardinality(String) NOT NULL,
    confidence Float64 DEFAULT 1.0,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_MEMORY_SHARING = """
CREATE TABLE IF NOT EXISTS memory_sharing (
    id UUID DEFAULT generateUUIDv4(),
    memory_id UUID NOT NULL,
    peer_id UUID NOT NULL,
    permission LowCardinality(String) DEFAULT 'SHARED',
    granted_by Nullable(UUID),
    granted_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID DEFAULT generateUUIDv4(),
    peer_id Nullable(UUID),
    preferences JSON DEFAULT '{}',
    communication_style String DEFAULT 'neutral',
    expertise_level LowCardinality(String) DEFAULT 'intermediate',
    -- chDB rejects JSON DEFAULT '[]' (only accepts '{}' as a JSON literal
    -- default). App code ensures dialectic_notes is set to '[]' on insert
    -- when not provided.
    dialectic_notes JSON,
    warmed_up Bool DEFAULT false,
    session_count Int32 DEFAULT 0,
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    updated_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_TRUST_ADJUSTMENTS = """
CREATE TABLE IF NOT EXISTS trust_adjustments (
    id UUID DEFAULT generateUUIDv4(),
    memory_id UUID NOT NULL,
    old_score Float64 NOT NULL,
    new_score Float64 NOT NULL,
    signal LowCardinality(String) NOT NULL,
    adjustment_amount Float64 NOT NULL,
    reason Nullable(String),
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_THOUGHT_CHAINS = """
CREATE TABLE IF NOT EXISTS thought_chains (
    id UUID DEFAULT generateUUIDv4(),
    session_id Nullable(String),
    parent_chain_id Nullable(UUID),
    status LowCardinality(String) DEFAULT 'active',
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    updated_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_THOUGHTS = """
CREATE TABLE IF NOT EXISTS thoughts (
    id UUID DEFAULT generateUUIDv4(),
    chain_id UUID NOT NULL,
    thought String NOT NULL,
    thought_number Int32 NOT NULL,
    total_thoughts Int32 NOT NULL,
    next_thought_needed Bool NOT NULL,
    is_revision Bool DEFAULT false,
    revises_thought_id Nullable(UUID),
    branch_from_thought_id Nullable(UUID),
    branch_id Nullable(String),
    -- chDB doesn't allow Nullable(Array(T)). Empty array = no embedding.
    embedding Array(Float32),
    content_hash Nullable(String),
    memory_id Nullable(UUID),
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""

SQL_CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID DEFAULT generateUUIDv4(),
    event_type LowCardinality(String) NOT NULL,
    severity LowCardinality(String) DEFAULT 'info',
    session_id Nullable(UUID),
    peer_id Nullable(String),
    agent_name Nullable(String),
    tool_name Nullable(String),
    memory_id Nullable(UUID),
    description Nullable(String),
    details Nullable(JSON),
    state_before Nullable(JSON),
    state_after Nullable(JSON),
    ip_address Nullable(IPv4),
    created_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    occurred_at DateTime64(9, 'UTC') DEFAULT now64(9, 'UTC'),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY id
"""


# =============================================================================
# CREATE INDEX statements (added after tables exist)
# =============================================================================
# These are minmax / set skip indexes. They DO NOT speed up the
# cosineDistance() brute-force search; they speed up the WHERE
# pre-filter (e.g. WHERE is_archived = false) that runs before the
# distance computation. With 80 memories this matters very little;
# at 100K+ it's the main lever for keeping query latency low.

# Memories: minmax on trust_score and last_accessed_at, with WHERE
# predicates to make them partial indexes. chDB supports `WHERE` on
# skip indexes; we replicate the Postgres partial-index behavior.

SQL_INDEX_MEMORIES_TRUST = """
ALTER TABLE memories
ADD INDEX IF NOT EXISTS idx_memories_trust trust_score
TYPE minmax(trust_score) GRANULARITY 4
"""
# Note: chDB doesn't support `WHERE NOT is_archived` directly on
# skip-index DDL in 4.2.1. The WHERE filter happens at query time,
# not at index build time. The minmax index covers all rows; the
# WHERE clause is applied during the scan. Same effective behavior
# as Postgres partial index, just with more rows in the index.

SQL_INDEX_MEMORIES_LAST_ACCESSED = """
ALTER TABLE memories
ADD INDEX IF NOT EXISTS idx_memories_last_accessed last_accessed_at
TYPE minmax(last_accessed_at) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_PEER = """
ALTER TABLE memories
ADD INDEX IF NOT EXISTS idx_memories_peer peer_id
TYPE minmax(peer_id) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_EMBEDDING_MODEL = """
ALTER TABLE memories
ADD INDEX IF NOT EXISTS idx_memories_embedding_model embedding_model
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_1024_MEMORY_ID = """
ALTER TABLE memories_1024
ADD INDEX IF NOT EXISTS idx_memories_1024_memory_id memory_id
TYPE minmax(memory_id) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_1024_TRUST = """
ALTER TABLE memories_1024
ADD INDEX IF NOT EXISTS idx_memories_1024_trust trust_score
TYPE minmax(trust_score) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_1024_ELEVATED_AT = """
ALTER TABLE memories_1024
ADD INDEX IF NOT EXISTS idx_memories_1024_elevated_at elevated_at
TYPE minmax(elevated_at) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_IMAGE_MEMORY_ID = """
ALTER TABLE memories_image
ADD INDEX IF NOT EXISTS idx_memories_image_memory_id memory_id
TYPE minmax(memory_id) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_IMAGE_SHA256 = """
ALTER TABLE memories_image
ADD INDEX IF NOT EXISTS idx_memories_image_sha256 image_sha256
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_IMAGE_TRUST = """
ALTER TABLE memories_image
ADD INDEX IF NOT EXISTS idx_memories_image_trust trust_score
TYPE minmax(trust_score) GRANULARITY 4
"""

SQL_INDEX_MEMORIES_IMAGE_CREATED_AT = """
ALTER TABLE memories_image
ADD INDEX IF NOT EXISTS idx_memories_image_created_at created_at
TYPE minmax(created_at) GRANULARITY 4
"""

SQL_INDEX_MEMORY_RELATIONSHIPS_SOURCE = """
ALTER TABLE memory_relationships
ADD INDEX IF NOT EXISTS idx_mem_rel_source source_id
TYPE minmax(source_id) GRANULARITY 4
"""

SQL_INDEX_MEMORY_RELATIONSHIPS_TARGET = """
ALTER TABLE memory_relationships
ADD INDEX IF NOT EXISTS idx_mem_rel_target target_id
TYPE minmax(target_id) GRANULARITY 4
"""

SQL_INDEX_MEMORY_RELATIONSHIPS_TYPE = """
ALTER TABLE memory_relationships
ADD INDEX IF NOT EXISTS idx_mem_rel_type relationship_type
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_ENTITIES_PEER = """
ALTER TABLE entities
ADD INDEX IF NOT EXISTS idx_entities_peer peer_id
TYPE minmax(peer_id) GRANULARITY 4
"""

SQL_INDEX_ENTITIES_TYPE = """
ALTER TABLE entities
ADD INDEX IF NOT EXISTS idx_entities_type entity_type
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_ENTITY_RELATIONSHIPS_SOURCE = """
ALTER TABLE entity_relationships
ADD INDEX IF NOT EXISTS idx_ent_rel_source source_entity_id
TYPE minmax(source_entity_id) GRANULARITY 4
"""

SQL_INDEX_ENTITY_RELATIONSHIPS_TARGET = """
ALTER TABLE entity_relationships
ADD INDEX IF NOT EXISTS idx_ent_rel_target target_entity_id
TYPE minmax(target_entity_id) GRANULARITY 4
"""

SQL_INDEX_MEMORY_SHARING_MEMORY = """
ALTER TABLE memory_sharing
ADD INDEX IF NOT EXISTS idx_mem_sharing_memory memory_id
TYPE minmax(memory_id) GRANULARITY 4
"""

SQL_INDEX_MEMORY_SHARING_PEER = """
ALTER TABLE memory_sharing
ADD INDEX IF NOT EXISTS idx_mem_sharing_peer peer_id
TYPE minmax(peer_id) GRANULARITY 4
"""

SQL_INDEX_USER_PROFILES_PEER = """
ALTER TABLE user_profiles
ADD INDEX IF NOT EXISTS idx_user_profiles_peer peer_id
TYPE minmax(peer_id) GRANULARITY 4
"""

SQL_INDEX_TRUST_ADJ_MEMORY = """
ALTER TABLE trust_adjustments
ADD INDEX IF NOT EXISTS idx_trust_adj_memory memory_id
TYPE minmax(memory_id) GRANULARITY 4
"""

SQL_INDEX_TRUST_ADJ_CREATED = """
ALTER TABLE trust_adjustments
ADD INDEX IF NOT EXISTS idx_trust_adj_created created_at
TYPE minmax(created_at) GRANULARITY 4
"""

SQL_INDEX_THOUGHT_CHAINS_SESSION = """
ALTER TABLE thought_chains
ADD INDEX IF NOT EXISTS idx_thought_chains_session session_id
TYPE minmax(session_id) GRANULARITY 4
"""

SQL_INDEX_THOUGHT_CHAINS_PARENT = """
ALTER TABLE thought_chains
ADD INDEX IF NOT EXISTS idx_thought_chains_parent parent_chain_id
TYPE minmax(parent_chain_id) GRANULARITY 4
"""

SQL_INDEX_THOUGHT_CHAINS_STATUS = """
ALTER TABLE thought_chains
ADD INDEX IF NOT EXISTS idx_thought_chains_status status
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_THOUGHTS_CHAIN = """
ALTER TABLE thoughts
ADD INDEX IF NOT EXISTS idx_thoughts_chain chain_id
TYPE minmax(chain_id) GRANULARITY 4
"""

SQL_INDEX_THOUGHTS_BRANCH = """
ALTER TABLE thoughts
ADD INDEX IF NOT EXISTS idx_thoughts_branch branch_id
TYPE minmax(branch_id) GRANULARITY 4
"""

SQL_INDEX_THOUGHTS_REVISES = """
ALTER TABLE thoughts
ADD INDEX IF NOT EXISTS idx_thoughts_revises revises_thought_id
TYPE minmax(revises_thought_id) GRANULARITY 4
"""

SQL_INDEX_THOUGHTS_MEMORY = """
ALTER TABLE thoughts
ADD INDEX IF NOT EXISTS idx_thoughts_memory memory_id
TYPE minmax(memory_id) GRANULARITY 4
"""

SQL_INDEX_AUDIT_OCCURRED_AT = """
ALTER TABLE audit_log
ADD INDEX IF NOT EXISTS idx_audit_log_occurred_at occurred_at
TYPE minmax(occurred_at) GRANULARITY 4
"""

SQL_INDEX_AUDIT_EVENT_TYPE = """
ALTER TABLE audit_log
ADD INDEX IF NOT EXISTS idx_audit_log_event_type event_type
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_AUDIT_SEVERITY = """
ALTER TABLE audit_log
ADD INDEX IF NOT EXISTS idx_audit_log_severity severity
TYPE set(0) GRANULARITY 4
"""

SQL_INDEX_AUDIT_SESSION_ID = """
ALTER TABLE audit_log
ADD INDEX IF NOT EXISTS idx_audit_log_session_id session_id
TYPE minmax(session_id) GRANULARITY 4
"""

SQL_INDEX_AUDIT_CREATED_AT = """
ALTER TABLE audit_log
ADD INDEX IF NOT EXISTS idx_audit_log_created_at created_at
TYPE minmax(created_at) GRANULARITY 4
"""


# =============================================================================
# Aggregated helpers
# =============================================================================

# All CREATE TABLE statements in dependency order. Even though chDB
# doesn't enforce FKs, this order matches the natural dependency graph
# and matches the Postgres schema for readability.
CREATE_TABLES_IN_ORDER: list[str] = [
    SQL_CREATE_PEERS,
    SQL_CREATE_MEMORIES,
    SQL_CREATE_MEMORIES_1024,
    SQL_CREATE_MEMORIES_IMAGE,
    SQL_CREATE_MEMORY_RELATIONSHIPS,
    SQL_CREATE_ENTITIES,
    SQL_CREATE_ENTITY_RELATIONSHIPS,
    SQL_CREATE_MEMORY_SHARING,
    SQL_CREATE_USER_PROFILES,
    SQL_CREATE_TRUST_ADJUSTMENTS,
    SQL_CREATE_THOUGHT_CHAINS,
    SQL_CREATE_THOUGHTS,
    SQL_CREATE_AUDIT_LOG,
]

# All ALTER TABLE ADD INDEX statements. Run after all CREATE TABLEs.
CREATE_INDEXES_IN_ORDER: list[str] = [
    SQL_INDEX_MEMORIES_TRUST,
    SQL_INDEX_MEMORIES_LAST_ACCESSED,
    SQL_INDEX_MEMORIES_PEER,
    SQL_INDEX_MEMORIES_EMBEDDING_MODEL,
    SQL_INDEX_MEMORIES_1024_MEMORY_ID,
    SQL_INDEX_MEMORIES_1024_TRUST,
    SQL_INDEX_MEMORIES_1024_ELEVATED_AT,
    SQL_INDEX_MEMORIES_IMAGE_MEMORY_ID,
    SQL_INDEX_MEMORIES_IMAGE_SHA256,
    SQL_INDEX_MEMORIES_IMAGE_TRUST,
    SQL_INDEX_MEMORIES_IMAGE_CREATED_AT,
    SQL_INDEX_MEMORY_RELATIONSHIPS_SOURCE,
    SQL_INDEX_MEMORY_RELATIONSHIPS_TARGET,
    SQL_INDEX_MEMORY_RELATIONSHIPS_TYPE,
    SQL_INDEX_ENTITIES_PEER,
    SQL_INDEX_ENTITIES_TYPE,
    SQL_INDEX_ENTITY_RELATIONSHIPS_SOURCE,
    SQL_INDEX_ENTITY_RELATIONSHIPS_TARGET,
    SQL_INDEX_MEMORY_SHARING_MEMORY,
    SQL_INDEX_MEMORY_SHARING_PEER,
    SQL_INDEX_USER_PROFILES_PEER,
    SQL_INDEX_TRUST_ADJ_MEMORY,
    SQL_INDEX_TRUST_ADJ_CREATED,
    SQL_INDEX_THOUGHT_CHAINS_SESSION,
    SQL_INDEX_THOUGHT_CHAINS_PARENT,
    SQL_INDEX_THOUGHT_CHAINS_STATUS,
    SQL_INDEX_THOUGHTS_CHAIN,
    SQL_INDEX_THOUGHTS_BRANCH,
    SQL_INDEX_THOUGHTS_REVISES,
    SQL_INDEX_THOUGHTS_MEMORY,
    SQL_INDEX_AUDIT_OCCURRED_AT,
    SQL_INDEX_AUDIT_EVENT_TYPE,
    SQL_INDEX_AUDIT_SEVERITY,
    SQL_INDEX_AUDIT_SESSION_ID,
    SQL_INDEX_AUDIT_CREATED_AT,
]


def get_schema_sql() -> str:
    """Return all CREATE TABLE + CREATE INDEX SQL as a single concatenated string.

    Useful for one-shot script execution (e.g. ``chdb < schema.sql``).
    """
    return "\n".join(
        ["-- === CREATE TABLES ==="]
        + CREATE_TABLES_IN_ORDER
        + [""]
        + ["-- === CREATE INDEXES ==="]
        + CREATE_INDEXES_IN_ORDER
    )


__all__ = [
    # Constants
    "TABLE_PEERS",
    "TABLE_MEMORIES",
    "TABLE_MEMORIES_1024",
    "TABLE_MEMORIES_IMAGE",
    "TABLE_MEMORY_RELATIONSHIPS",
    "TABLE_ENTITIES",
    "TABLE_ENTITY_RELATIONSHIPS",
    "TABLE_MEMORY_SHARING",
    "TABLE_USER_PROFILES",
    "TABLE_TRUST_ADJUSTMENTS",
    "TABLE_THOUGHT_CHAINS",
    "TABLE_THOUGHTS",
    "TABLE_AUDIT_LOG",
    "ALL_TABLES",
    "VALID_SOURCE_TYPES",
    "VALID_PEER_ROLES",
    "VALID_MEMORY_RELATIONSHIPS",
    "VALID_ENTITY_TYPES",
    "VALID_PERMISSIONS",
    "VALID_TRUST_SIGNALS",
    "VALID_CHAIN_STATUSES",
    "VALID_AUDIT_EVENT_TYPES",
    "VALID_AUDIT_SEVERITIES",
    # Aggregated
    "CREATE_TABLES_IN_ORDER",
    "CREATE_INDEXES_IN_ORDER",
    "get_schema_sql",
]
