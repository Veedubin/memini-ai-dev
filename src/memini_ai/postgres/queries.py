"""Reusable SQL query builders for pgvector/pgvectorscale backend.

This module provides parameterized SQL queries using asyncpg's $1, $2 notation.
All queries are ready for use with asyncpg's conn.fetch() and conn.execute() methods.

Query Categories:
- Vector Search: Cosine distance similarity search
- CRUD Operations: Insert, get, update, delete for memories
- Relationships: Memory relationship queries
- Trust Engine: Trust score updates and decay operations
"""

# =============================================================================
# Vector Search Queries
# =============================================================================

SEARCH_MEMORIES_VECTOR = """
SELECT id, text, source_type, trust_score, retrieval_count, is_archived, metadata,
       embedding <=> $1 as distance
FROM memories
WHERE embedding <=> $1 < $2
AND is_archived = FALSE
ORDER BY embedding <=> $1
LIMIT $3
"""

SEARCH_MEMORIES_WITH_PEER = """
SELECT DISTINCT m.id, m.text, m.source_type, m.trust_score, m.retrieval_count, m.is_archived, m.metadata
FROM memories m
JOIN memory_sharing ms ON m.id = ms.memory_id
WHERE ms.peer_id = $1
AND ms.permission IN ('SHARED', 'INHERITED')
AND m.is_archived = FALSE
ORDER BY m.retrieval_count DESC
LIMIT $2
"""

# =============================================================================
# Memory CRUD Queries
# =============================================================================

INSERT_MEMORY = """
INSERT INTO memories (id, text, embedding, source_type, content_hash, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
RETURNING id
"""

GET_MEMORY_BY_ID = """
SELECT id, text, embedding, source_type, content_hash, metadata,
       trust_score, retrieval_count, is_archived, last_accessed_at,
       created_at, updated_at
FROM memories
WHERE id = $1 AND is_archived = FALSE
"""

UPDATE_MEMORY_TEXT = """
UPDATE memories
SET text = $2, updated_at = NOW()
WHERE id = $1 AND is_archived = FALSE
RETURNING id
"""

UPDATE_MEMORY_METADATA = """
UPDATE memories
SET metadata = $2, updated_at = NOW()
WHERE id = $1 AND is_archived = FALSE
RETURNING id
"""

DELETE_MEMORY = """
UPDATE memories
SET is_archived = TRUE, updated_at = NOW()
WHERE id = $1
RETURNING id
"""

# =============================================================================
# Trust Engine Queries
# =============================================================================

UPDATE_TRUST_SCORE = """
UPDATE memories
SET trust_score = $1, last_accessed_at = NOW()
WHERE id = $2
RETURNING id, trust_score
"""

INCREMENT_RETRIEVAL_COUNT = """
UPDATE memories
SET retrieval_count = retrieval_count + 1, last_accessed_at = NOW()
WHERE id = $1
RETURNING id, retrieval_count
"""

TRUST_DECAY_BULK = """
UPDATE memories
SET trust_score = trust_score * POWER(0.5, (
    EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 86400.0 * 0.1 / 90
  ))
WHERE is_archived = FALSE
AND trust_score > 0.1
AND last_accessed_at < NOW() - INTERVAL '7 days'
"""

ARCHIVE_LOW_TRUST = """
UPDATE memories
SET is_archived = TRUE, updated_at = NOW()
WHERE trust_score < $1 AND is_archived = FALSE
"""

# =============================================================================
# Relationship Queries
# =============================================================================

GET_MEMORY_RELATIONSHIPS = """
SELECT id, source_id, target_id, relationship_type, confidence, created_at, metadata
FROM memory_relationships
WHERE source_id = $1 OR target_id = $1
"""

GET_RELATIONSHIPS_BY_TYPE = """
SELECT id, source_id, target_id, relationship_type, confidence, created_at, metadata
FROM memory_relationships
WHERE (source_id = $1 OR target_id = $1)
AND relationship_type = $2
"""

INSERT_RELATIONSHIP = """
INSERT INTO memory_relationships (id, source_id, target_id, relationship_type, confidence, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (source_id, target_id, relationship_type) DO UPDATE
SET confidence = $5, metadata = $6
RETURNING id
"""

DELETE_RELATIONSHIP = """
DELETE FROM memory_relationships
WHERE id = $1
RETURNING id
"""

GET_CONTRADICTIONS_LOW_TRUST = """
SELECT
    m1.id AS memory_a_id,
    m1.text AS memory_a_text,
    m1.trust_score AS trust_a,
    m2.id AS memory_b_id,
    m2.text AS memory_b_text,
    m2.trust_score AS trust_b
FROM memory_relationships r
JOIN memories m1 ON r.source_id = m1.id
JOIN memories m2 ON r.target_id = m2.id
WHERE r.relationship_type = 'CONTRADICTS'
AND m1.is_archived = FALSE
AND m2.is_archived = FALSE
AND m1.trust_score < 0.5
AND m2.trust_score < 0.5
ORDER BY (m1.trust_score + m2.trust_score) ASC
"""

# =============================================================================
# Entity Queries
# =============================================================================

INSERT_ENTITY = """
INSERT INTO entities (id, name, entity_type, metadata)
VALUES ($1, $2, $3, $4)
RETURNING id
"""

GET_ENTITY_BY_ID = """
SELECT id, name, entity_type, metadata, created_at
FROM entities
WHERE id = $1
"""

GET_ENTITIES_BY_TYPE = """
SELECT id, name, entity_type, metadata, created_at
FROM entities
WHERE entity_type = $1
"""

INSERT_ENTITY_RELATIONSHIP = """
INSERT INTO entity_relationships (id, source_entity_id, target_entity_id, relationship_type, metadata)
VALUES ($1, $2, $3, $4, $5)
RETURNING id
"""

GET_ENTITY_RELATIONSHIPS = """
SELECT id, source_entity_id, target_entity_id, relationship_type, created_at, metadata
FROM entity_relationships
WHERE source_entity_id = $1 OR target_entity_id = $1
"""

# =============================================================================
# Multi-Peer Sharing Queries
# =============================================================================

INSERT_MEMORY_SHARING = """
INSERT INTO memory_sharing (id, memory_id, peer_id, permission)
VALUES ($1, $2, $3, $4)
RETURNING id
"""

GET_SHARED_MEMORIES = """
SELECT m.id, m.text, m.source_type, m.metadata, ms.permission, ms.created_at
FROM memories m
JOIN memory_sharing ms ON m.id = ms.memory_id
WHERE ms.peer_id = $1
AND ms.permission IN ('SHARED', 'INHERITED')
AND m.is_archived = FALSE
ORDER BY ms.created_at DESC
LIMIT $2
"""

REVOKE_SHARING = """
DELETE FROM memory_sharing
WHERE memory_id = $1 AND peer_id = $2
RETURNING id
"""

# =============================================================================
# Trust Adjustments (Audit Log)
# =============================================================================

INSERT_TRUST_ADJUSTMENT = """
INSERT INTO trust_adjustments (id, memory_id, adjustment, reason, adjusted_by)
VALUES ($1, $2, $3, $4, $5)
RETURNING id
"""

GET_TRUST_ADJUSTMENTS = """
SELECT id, memory_id, adjustment, reason, adjusted_by, created_at
FROM trust_adjustments
WHERE memory_id = $1
ORDER BY created_at DESC
"""

# =============================================================================
# Index Management
# =============================================================================

GET_MEMORY_COUNT = """
SELECT COUNT(*) as total,
       COUNT(*) FILTER (WHERE is_archived = FALSE) as active,
       COUNT(*) FILTER (WHERE is_archived = TRUE) as archived
FROM memories
"""

GET_LOW_TRUST_MEMORIES = """
SELECT id, text, trust_score, last_accessed_at, is_archived
FROM memories
WHERE trust_score < $1 AND is_archived = FALSE
ORDER BY trust_score ASC
LIMIT $2
"""
