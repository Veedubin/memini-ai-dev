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
       embedding,
       embedding <=> $1::vector as distance
FROM memories
WHERE embedding <=> $1::vector < $2
AND is_archived = FALSE
ORDER BY embedding <=> $1::vector
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
# Entity Queries (Knowledge Graph)
# =============================================================================

# Entity CRUD
INSERT_ENTITY = """
INSERT INTO entities (id, name, entity_type, canonical_name, confidence, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (name, entity_type) DO UPDATE
SET canonical_name = EXCLUDED.canonical_name,
    confidence = EXCLUDED.confidence,
    mention_count = entities.mention_count + 1,
    last_seen_at = NOW()
RETURNING id
"""

UPSERT_ENTITY = """
INSERT INTO entities (id, name, entity_type, canonical_name, confidence, mention_count, last_seen_at, metadata)
VALUES ($1, $2, $3, $4, $5, 1, NOW(), $6)
ON CONFLICT (name, entity_type) DO UPDATE
SET canonical_name = EXCLUDED.canonical_name,
    confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
    mention_count = entities.mention_count + 1,
    last_seen_at = NOW()
RETURNING id
"""

GET_ENTITIES = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
ORDER BY mention_count DESC
LIMIT $1
"""

GET_ENTITIES_WITH_RELATIONSHIPS = """
SELECT
    e.id, e.name, e.entity_type, e.canonical_name, e.confidence,
    er.target_entity_id, er.relationship_type, er.confidence as rel_confidence
FROM entities e
LEFT JOIN entity_relationships er ON e.id = er.source_entity_id
ORDER BY e.mention_count DESC
LIMIT $1
"""

GET_ENTITY_BY_ID = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
WHERE id = $1
"""

GET_ENTITIES_BY_TYPE = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
WHERE entity_type = $1
ORDER BY mention_count DESC
LIMIT $2
"""

UPDATE_ENTITY_METADATA = """
UPDATE entities
SET metadata = $2, last_seen_at = NOW()
WHERE id = $1
RETURNING id
"""

DELETE_ENTITY = """
DELETE FROM entities WHERE id = $1
RETURNING id
"""

# Entity Relationships
INSERT_ENTITY_RELATIONSHIP = """
INSERT INTO entity_relationships (id, source_entity_id, target_entity_id, relationship_type, confidence)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO UPDATE
SET confidence = EXCLUDED.confidence
RETURNING id
"""

GET_ENTITY_RELATIONSHIPS = """
SELECT id, source_entity_id, target_entity_id, relationship_type, confidence, created_at
FROM entity_relationships
WHERE source_entity_id = $1 OR target_entity_id = $1
"""

GET_ALL_ENTITY_RELATIONSHIPS = """
SELECT id, source_entity_id, target_entity_id, relationship_type, confidence, created_at
FROM entity_relationships
"""

DELETE_ENTITY_RELATIONSHIP = """
DELETE FROM entity_relationships
WHERE id = $1
RETURNING id
"""

# Entity Statistics
GET_ENTITY_STATS = """
SELECT
    COUNT(*) as total_entities,
    COUNT(*) FILTER (WHERE entity_type = 'PERSON') as persons,
    COUNT(*) FILTER (WHERE entity_type = 'ORGANIZATION') as organizations,
    COUNT(*) FILTER (WHERE entity_type = 'CONCEPT') as concepts,
    COUNT(*) FILTER (WHERE entity_type = 'CODE') as codes,
    COUNT(*) FILTER (WHERE entity_type = 'PROJECT') as projects,
    COUNT(*) FILTER (WHERE entity_type = 'LOCATION') as locations,
    COUNT(*) FILTER (WHERE entity_type = 'UNKNOWN') as unknowns,
    COUNT(*) FILTER (WHERE mention_count > 1) as entities_with_mentions
FROM entities
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

# =============================================================================
# Thought Chain Queries (Phase 5)
# =============================================================================

# Thought Chain CRUD
INSERT_THOUGHT_CHAIN = """
INSERT INTO thought_chains (id, session_id, parent_chain_id, status)
VALUES ($1, $2, $3, $4)
RETURNING id, session_id, parent_chain_id, status, created_at, updated_at
"""

GET_THOUGHT_CHAIN_BY_ID = """
SELECT id, session_id, parent_chain_id, status, created_at, updated_at
FROM thought_chains
WHERE id = $1
"""

UPDATE_THOUGHT_CHAIN_STATUS = """
UPDATE thought_chains
SET status = $2, updated_at = NOW()
WHERE id = $1
RETURNING id, session_id, parent_chain_id, status, created_at, updated_at
"""

GET_THOUGHT_CHAINS_BY_SESSION = """
SELECT id, session_id, parent_chain_id, status, created_at, updated_at
FROM thought_chains
WHERE session_id = $1
ORDER BY created_at DESC
"""

# Thought CRUD
INSERT_THOUGHT = """
INSERT INTO thoughts (id, chain_id, thought, thought_number, total_thoughts,
                      next_thought_needed, is_revision, revises_thought_id,
                      branch_from_thought_id, branch_id, embedding, content_hash, memory_id)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
RETURNING id, chain_id, thought, thought_number, total_thoughts,
           next_thought_needed, is_revision, revises_thought_id,
           branch_from_thought_id, branch_id, content_hash, memory_id, created_at
"""

GET_THOUGHTS_BY_CHAIN = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = $1
ORDER BY thought_number ASC, created_at ASC
"""

GET_THOUGHT_BY_NUMBER = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = $1 AND thought_number = $2
ORDER BY created_at DESC
LIMIT 1
"""

GET_LAST_THOUGHT_IN_CHAIN = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = $1
ORDER BY thought_number DESC, created_at DESC
LIMIT 1
"""

GET_THOUGHT_BRANCHES = """
SELECT DISTINCT branch_id
FROM thoughts
WHERE chain_id = $1 AND branch_id IS NOT NULL
ORDER BY branch_id
"""

COUNT_THOUGHTS_IN_CHAIN = """
SELECT COUNT(*) as thought_count
FROM thoughts
WHERE chain_id = $1
"""

# Semantic search across thought chains
SEARCH_THOUGHT_CHAINS_BY_EMBEDDING = """
WITH ranked_thoughts AS (
    SELECT
        t.id,
        t.chain_id,
        t.thought,
        t.thought_number,
        t.branch_id,
        t.embedding <=> $1::vector as distance
    FROM thoughts t
    JOIN thought_chains tc ON t.chain_id = tc.id
    WHERE t.embedding IS NOT NULL
    AND tc.status = 'active'
    ORDER BY t.embedding <=> $1::vector
    LIMIT $2
)
SELECT
    rt.chain_id,
    tc.session_id,
    rt.thought as snippet,
    rt.distance as score,
    (SELECT COUNT(*) FROM thoughts WHERE chain_id = rt.chain_id) as thought_count
FROM ranked_thoughts rt
JOIN thought_chains tc ON rt.chain_id = tc.id
GROUP BY rt.chain_id, tc.session_id, rt.thought, rt.distance, rt.thought_count
ORDER BY MIN(rt.distance) ASC
LIMIT $3
"""

UPDATE_THOUGHT_MEMORY_ID = """
UPDATE thoughts
SET memory_id = $2
WHERE id = $1
"""
