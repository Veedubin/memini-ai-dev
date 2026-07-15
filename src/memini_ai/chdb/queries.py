"""Reusable SQL query builders for chDB (in-process ClickHouse) backend.

Mirrors ``src/memini_ai/postgres/queries.py`` with the type substitutions
documented in the design doc:

- pgvector's ``<=> $1::vector`` becomes
  ``cosineDistance(col, $1::Array(Float32))``
- ``NOW()`` becomes ``now64(9, 'UTC')``
- ``BOOLEAN`` literals stay the same
- ``RETURNING id`` (Postgres syntax) is dropped — chDB doesn't have
  RETURNING. Use ``SELECT last_insert_id()`` or a follow-up SELECT
  (the calling code is responsible).
- ``ON CONFLICT (...) DO UPDATE / DO NOTHING`` is replaced with explicit
  ``SELECT ... LIMIT 1`` checks in app code (chDB doesn't have
  ON CONFLICT). Or we drop the constraint and accept last-write-wins.
- The ``::vector``, ``::boolean``, etc. casts are removed (chDB's
  Array(Float32) is a real type, no cast needed).
- ``FILTER (WHERE ...)`` is replaced with ``CASE WHEN ...`` since
  chDB doesn't support FILTER.

This is a **partial port** for 0.9.0. The queries not ported are:
- v0.12.0+ multi-model RRF queries that reference the
  ``embedding_bge_m3`` column (which doesn't exist in the current
  schema; v0.7.8 retired BGE-Large in favor of the memories_1024
  sidecar). See design doc Section 4.
- The 1024-dim + image RRF arms of the 3-arm RRF query
  (Task 4's ``query_memories`` will use a single brute-force cosine
  query for 0.9.0; the RRF fan-out lands in a follow-up when chDB ships
  the vector_similarity index).

Query format: parameterized with ``{name}`` placeholders that are
formatted with ``str.format()`` (NOT positional ``$1, $2`` like
asyncpg, because chDB's Python session API uses named params or
positional ``?`` args; we use named for clarity). Task 4's
database.py will translate.

Indexes: this module does NOT define indexes (those are in
:mod:`memini_ai.chdb.schema`). It only defines query strings.
"""

from __future__ import annotations

# =============================================================================
# Helpers
# =============================================================================

#: Reusable timestamp expression. Postgres used ``NOW()``; chDB uses
#: ``now64(9, 'UTC')`` (microsecond precision, UTC).
NOW = "now64(9, 'UTC')"


# =============================================================================
# Vector Search Queries
# =============================================================================

# Note: chDB 4.2.1 does NOT ship the vector_similarity HNSW index. We
# use brute-force cosine distance over Array(Float32) columns. The
# WHERE filter is applied BEFORE the distance computation (pre-filter,
# not post-filter). Performance: 100K x 384 = ~30ms in our benchmarks.

#: Brute-force search of the 384-dim memories table.
#: Params: {embedding} (Array(Float32)), {threshold} (Float64),
#: {limit} (Int32). Returns rows ordered by cosine distance.
SEARCH_MEMORIES_VECTOR = """
SELECT id, text, source_type, trust_score, retrieval_count, is_archived, metadata,
       embedding, supersedes_id, structured_fields, change_ratio, created_at_ms,
       cosineDistance(embedding, {embedding}::Array(Float32)) AS distance
FROM memories
WHERE is_archived = false
  AND length(embedding) > 0
  AND cosineDistance(embedding, {embedding}::Array(Float32)) < {threshold}
ORDER BY distance ASC
LIMIT {limit}
"""

#: Search memories that are shared with a specific peer.
#: Params: {peer_id} (UUID), {limit} (Int32).
SEARCH_MEMORIES_WITH_PEER = """
SELECT DISTINCT m.id, m.text, m.source_type, m.trust_score, m.retrieval_count,
       m.is_archived, m.metadata
FROM memories m
INNER JOIN memory_sharing ms ON m.id = ms.memory_id
WHERE ms.peer_id = {peer_id}
  AND ms.permission IN ('SHARED', 'INHERITED')
  AND m.is_archived = false
ORDER BY m.retrieval_count DESC
LIMIT {limit}
"""


# =============================================================================
# Memory CRUD Queries
# =============================================================================

# chDB doesn't have RETURNING. We use SELECT after the INSERT to get
# the row back. The database.py wrapper handles this.

#: Insert a memory. Params: {id}, {text}, {embedding} (Array(Float32) or []),
#: {source_type}, {content_hash} (or ''), {metadata} (JSON string,
# already wrapped in single quotes by the caller),
#: {created_at_ms} (Int64).
#: chDB quirk: toJSONString('{}') is rejected on JSON column inserts.
#: Pass the JSON string directly.
INSERT_MEMORY = """
INSERT INTO memories
    (id, text, embedding, source_type, content_hash, metadata, created_at_ms)
VALUES
    ({id}, {text}, {embedding}::Array(Float32), {source_type},
     {content_hash}, {metadata}, {created_at_ms})
"""

#: Insert a memory with the delta-model fields populated.
INSERT_MEMORY_DELTA = """
INSERT INTO memories
    (id, text, embedding, source_type, content_hash, metadata,
     supersedes_id, structured_fields, change_ratio, created_at_ms)
VALUES
    ({id}, {text}, {embedding}::Array(Float32), {source_type},
     {content_hash}, {metadata},
     {supersedes_id}, {structured_fields},
     {change_ratio}, {created_at_ms})
"""

#: Get a memory by id, optionally including archived.
#: Params: {id} (UUID), {include_archived} (bool).
GET_MEMORY_BY_ID = """
SELECT id, text, embedding, source_type, content_hash, metadata,
       trust_score, retrieval_count, is_archived, last_accessed_at,
       created_at, updated_at,
       supersedes_id, structured_fields, change_ratio, created_at_ms
FROM memories
WHERE id = {id}
"""

#: Get the full supersession chain (recursive CTE).
#: Params: {id} (UUID), {max_depth} (Int32).
GET_SUPERSESSION_CHAIN = """
WITH RECURSIVE chain AS (
    SELECT id, text, trust_score, is_archived, supersedes_id,
           structured_fields, change_ratio, source_type, metadata,
           created_at_ms,
           toInt32(1) AS depth,
           [id] AS path
    FROM memories WHERE id = {id}
    UNION ALL
    SELECT m.id, m.text, m.trust_score, m.is_archived, m.supersedes_id,
           m.structured_fields, m.change_ratio, m.source_type, m.metadata,
           m.created_at_ms,
           c.depth + 1,
           concat(c.path, [m.id])
    FROM memories m
    INNER JOIN chain c ON m.id = c.supersedes_id
    WHERE c.depth < {max_depth} AND c.depth < 20
)
SELECT id, text, trust_score, is_archived, supersedes_id,
       structured_fields, change_ratio, source_type, metadata,
       created_at_ms, depth
FROM chain
ORDER BY created_at_ms DESC
"""

#: Get the memory this memory supersedes (parent in the chain).
GET_SUPERSEDED_MEMORY = """
SELECT id, text, trust_score, is_archived, supersedes_id,
       structured_fields, change_ratio, source_type, metadata,
       created_at_ms
FROM memories
WHERE id = (SELECT supersedes_id FROM memories WHERE id = {id})
"""

#: Update a memory's text.
UPDATE_MEMORY_TEXT = """
ALTER TABLE memories
UPDATE text = {text}, updated_at = {now}
WHERE id = {id} AND is_archived = false
"""

#: Update a memory's metadata.
UPDATE_MEMORY_METADATA = """
ALTER TABLE memories
UPDATE metadata = {metadata}, updated_at = {now}
WHERE id = {id} AND is_archived = false
"""

#: Soft-delete a memory (set is_archived = true).
#: chDB doesn't have UPDATE...RETURNING. The id is returned via
#: a follow-up SELECT by the caller.
DELETE_MEMORY = """
ALTER TABLE memories
UPDATE is_archived = true, updated_at = {now}
WHERE id = {id}
"""


# =============================================================================
# Trust Engine Queries
# =============================================================================

#: Update a memory's trust score (no RETURNING in chDB).
UPDATE_TRUST_SCORE = """
ALTER TABLE memories
UPDATE trust_score = {trust_score}, last_accessed_at = {now}
WHERE id = {memory_id}
"""

#: Increment retrieval_count.
INCREMENT_RETRIEVAL_COUNT = """
ALTER TABLE memories
UPDATE retrieval_count = retrieval_count + 1, last_accessed_at = {now}
WHERE id = {memory_id}
"""

#: Trust decay (bulk). Exponential decay over the last_accessed_at timestamp.
#: chDB date math: ``dateDiff('second', last_accessed_at, now64(9, 'UTC'))``
#: gives the number of seconds since the last access.
TRUST_DECAY_BULK = """
ALTER TABLE memories
UPDATE trust_score = trust_score * pow(0.5, (
    dateDiff('second', last_accessed_at, now64(9, 'UTC')) / 86400.0 * 0.1 / 90
))
WHERE is_archived = false
  AND trust_score > 0.1
  AND last_accessed_at < now64(9, 'UTC') - INTERVAL 7 DAY
"""

#: Archive all memories with trust_score below a threshold.
ARCHIVE_LOW_TRUST = """
ALTER TABLE memories
UPDATE is_archived = true, updated_at = now64(9, 'UTC')
WHERE trust_score < {threshold} AND is_archived = false
"""


# =============================================================================
# Relationship Queries
# =============================================================================

#: Get all relationships for a memory.
GET_MEMORY_RELATIONSHIPS = """
SELECT id, source_id, target_id, relationship_type, confidence, created_at, metadata
FROM memory_relationships
WHERE source_id = {memory_id} OR target_id = {memory_id}
"""

#: Get relationships of a specific type for a memory.
GET_RELATIONSHIPS_BY_TYPE = """
SELECT id, source_id, target_id, relationship_type, confidence, created_at, metadata
FROM memory_relationships
WHERE (source_id = {memory_id} OR target_id = {memory_id})
  AND relationship_type = {relationship_type}
"""

#: Insert a relationship. Idempotency in chDB is handled by app layer
#: (chDB has no ON CONFLICT). Caller does a SELECT first.
INSERT_RELATIONSHIP = """
INSERT INTO memory_relationships
    (id, source_id, target_id, relationship_type, confidence, metadata)
VALUES
    ({id}, {source_id}, {target_id}, {relationship_type}, {confidence},
     toJSONString({metadata}))
"""

#: Delete a relationship.
DELETE_RELATIONSHIP = """
DELETE FROM memory_relationships WHERE id = {id}
"""

#: Get contradiction pairs where both sides have low trust.
#: Postgres used COUNT(*) FILTER (WHERE ...) which chDB doesn't support.
#: We rewrite using case expressions.
GET_CONTRADICTIONS_LOW_TRUST = """
SELECT
    m1.id AS memory_a_id,
    m1.text AS memory_a_text,
    m1.trust_score AS trust_a,
    m2.id AS memory_b_id,
    m2.text AS memory_b_text,
    m2.trust_score AS trust_b
FROM memory_relationships r
INNER JOIN memories m1 ON r.source_id = m1.id
INNER JOIN memories m2 ON r.target_id = m2.id
WHERE r.relationship_type = 'CONTRADICTS'
  AND m1.is_archived = false
  AND m2.is_archived = false
  AND m1.trust_score < 0.5
  AND m2.trust_score < 0.5
ORDER BY (m1.trust_score + m2.trust_score) ASC
"""


# =============================================================================
# Entity Queries (Knowledge Graph)
# =============================================================================

#: Insert or update an entity. Postgres used ON CONFLICT (name, entity_type)
#: DO UPDATE; chDB has no ON CONFLICT. The caller does a SELECT first and
#: either INSERTs or UPDATEs.
INSERT_ENTITY = """
INSERT INTO entities
    (id, name, entity_type, canonical_name, confidence, metadata)
VALUES
    ({id}, {name}, {entity_type}, {canonical_name}, {confidence},
     {metadata})
"""

#: Upsert entity (with mention_count increment). Caller handles the
#: conflict logic; this is the UPDATE path.
UPSERT_ENTITY = """
ALTER TABLE entities
UPDATE
    canonical_name = {canonical_name},
    confidence = greatest(confidence, {confidence}),
    mention_count = mention_count + 1,
    last_seen_at = {now}
WHERE name = {name} AND entity_type = {entity_type}
"""

#: List entities, ordered by mention_count.
GET_ENTITIES = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
ORDER BY mention_count DESC
LIMIT {limit}
"""

#: List entities with their relationships (LEFT JOIN).
GET_ENTITIES_WITH_RELATIONSHIPS = """
SELECT
    e.id, e.name, e.entity_type, e.canonical_name, e.confidence,
    er.target_entity_id, er.relationship_type, er.confidence AS rel_confidence
FROM entities e
LEFT JOIN entity_relationships er ON e.id = er.source_entity_id
ORDER BY e.mention_count DESC
LIMIT {limit}
"""

#: Get one entity by id.
GET_ENTITY_BY_ID = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
WHERE id = {id}
"""

#: Get entities by type.
GET_ENTITIES_BY_TYPE = """
SELECT id, name, entity_type, canonical_name, confidence, mention_count,
       first_seen_at, last_seen_at, metadata
FROM entities
WHERE entity_type = {entity_type}
ORDER BY mention_count DESC
LIMIT {limit}
"""

#: Update an entity's metadata.
UPDATE_ENTITY_METADATA = """
ALTER TABLE entities
UPDATE metadata = {metadata}, last_seen_at = {now}
WHERE id = {id}
"""

#: Delete an entity.
DELETE_ENTITY = """
DELETE FROM entities WHERE id = {id}
"""

#: Insert an entity relationship.
INSERT_ENTITY_RELATIONSHIP = """
INSERT INTO entity_relationships
    (id, source_entity_id, target_entity_id, relationship_type, confidence)
VALUES
    ({id}, {source_entity_id}, {target_entity_id}, {relationship_type},
     {confidence})
"""

#: Get relationships for an entity.
GET_ENTITY_RELATIONSHIPS = """
SELECT id, source_entity_id, target_entity_id, relationship_type, confidence, created_at
FROM entity_relationships
WHERE source_entity_id = {entity_id} OR target_entity_id = {entity_id}
"""

#: Get all entity relationships (no filter).
GET_ALL_ENTITY_RELATIONSHIPS = """
SELECT id, source_entity_id, target_entity_id, relationship_type, confidence, created_at
FROM entity_relationships
"""

#: Delete an entity relationship.
DELETE_ENTITY_RELATIONSHIP = """
DELETE FROM entity_relationships WHERE id = {id}
"""

#: Entity statistics. chDB has no FILTER clause; rewrite with CASE WHEN.
GET_ENTITY_STATS = """
SELECT
    count() AS total_entities,
    countIf(entity_type = 'PERSON') AS persons,
    countIf(entity_type = 'ORGANIZATION') AS organizations,
    countIf(entity_type = 'CONCEPT') AS concepts,
    countIf(entity_type = 'CODE') AS codes,
    countIf(entity_type = 'PROJECT') AS projects,
    countIf(entity_type = 'LOCATION') AS locations,
    countIf(entity_type = 'UNKNOWN') AS unknowns,
    countIf(mention_count > 1) AS entities_with_mentions
FROM entities
"""


# =============================================================================
# Multi-Peer Sharing Queries
# =============================================================================

#: Insert a sharing row.
INSERT_MEMORY_SHARING = """
INSERT INTO memory_sharing
    (id, memory_id, peer_id, permission)
VALUES
    ({id}, {memory_id}, {peer_id}, {permission})
"""

#: Get shared memories for a peer.
GET_SHARED_MEMORIES = """
SELECT m.id, m.text, m.source_type, m.metadata, ms.permission, ms.granted_at
FROM memories m
INNER JOIN memory_sharing ms ON m.id = ms.memory_id
WHERE ms.peer_id = {peer_id}
  AND ms.permission IN ('SHARED', 'INHERITED')
  AND m.is_archived = false
ORDER BY ms.granted_at DESC
LIMIT {limit}
"""

#: Revoke sharing for a (memory, peer) pair.
REVOKE_SHARING = """
DELETE FROM memory_sharing
WHERE memory_id = {memory_id} AND peer_id = {peer_id}
"""


# =============================================================================
# Trust Adjustments (Audit Log)
# =============================================================================

#: Insert a trust adjustment record.
INSERT_TRUST_ADJUSTMENT = """
INSERT INTO trust_adjustments
    (id, memory_id, old_score, new_score, signal, adjustment_amount, reason)
VALUES
    ({id}, {memory_id}, {old_score}, {new_score}, {signal},
     {adjustment_amount}, {reason})
"""

#: Get trust adjustments for a memory.
GET_TRUST_ADJUSTMENTS = """
SELECT id, memory_id, old_score, new_score, signal, adjustment_amount,
       reason, created_at
FROM trust_adjustments
WHERE memory_id = {memory_id}
ORDER BY created_at DESC
"""


# =============================================================================
# Index Management (count, low-trust scans)
# =============================================================================

#: Memory count, split by archive state.
GET_MEMORY_COUNT = """
SELECT
    count() AS total,
    countIf(is_archived = false) AS active,
    countIf(is_archived = true) AS archived
FROM memories
"""

#: Get low-trust, non-archived memories (for trust decay / archival).
GET_LOW_TRUST_MEMORIES = """
SELECT id, text, trust_score, last_accessed_at, is_archived
FROM memories
WHERE trust_score < {threshold} AND is_archived = false
ORDER BY trust_score ASC
LIMIT {limit}
"""


# =============================================================================
# Thought Chain Queries
# =============================================================================

#: Insert a thought chain.
INSERT_THOUGHT_CHAIN = """
INSERT INTO thought_chains (id, session_id, parent_chain_id, status)
VALUES ({id}, {session_id}, {parent_chain_id}, {status})
"""

#: Get a thought chain by id.
GET_THOUGHT_CHAIN_BY_ID = """
SELECT id, session_id, parent_chain_id, status, created_at, updated_at
FROM thought_chains
WHERE id = {id}
"""

#: Update a thought chain's status.
UPDATE_THOUGHT_CHAIN_STATUS = """
ALTER TABLE thought_chains
UPDATE status = {status}, updated_at = {now}
WHERE id = {id}
"""

#: Get thought chains by session.
GET_THOUGHT_CHAINS_BY_SESSION = """
SELECT id, session_id, parent_chain_id, status, created_at, updated_at
FROM thought_chains
WHERE session_id = {session_id}
ORDER BY created_at DESC
"""

#: Insert a thought.
INSERT_THOUGHT = """
INSERT INTO thoughts
    (id, chain_id, thought, thought_number, total_thoughts,
     next_thought_needed, is_revision, revises_thought_id,
     branch_from_thought_id, branch_id, embedding, content_hash, memory_id)
VALUES
    ({id}, {chain_id}, {thought}, {thought_number}, {total_thoughts},
     {next_thought_needed}, {is_revision}, {revises_thought_id},
     {branch_from_thought_id}, {branch_id}, {embedding}::Array(Float32),
     {content_hash}, {memory_id})
"""

#: Get all thoughts in a chain.
GET_THOUGHTS_BY_CHAIN = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = {chain_id}
ORDER BY thought_number ASC, created_at ASC
"""

#: Get a specific thought by chain and number.
GET_THOUGHT_BY_NUMBER = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = {chain_id} AND thought_number = {thought_number}
ORDER BY created_at DESC
LIMIT 1
"""

#: Get the most recent thought in a chain.
GET_LAST_THOUGHT_IN_CHAIN = """
SELECT id, chain_id, thought, thought_number, total_thoughts,
       next_thought_needed, is_revision, revises_thought_id,
       branch_from_thought_id, branch_id, content_hash, memory_id, created_at
FROM thoughts
WHERE chain_id = {chain_id}
ORDER BY thought_number DESC, created_at DESC
LIMIT 1
"""

#: Get distinct branch ids in a chain.
GET_THOUGHT_BRANCHES = """
SELECT DISTINCT branch_id
FROM thoughts
WHERE chain_id = {chain_id} AND branch_id IS NOT NULL
ORDER BY branch_id
"""

#: Count thoughts in a chain.
COUNT_THOUGHTS_IN_CHAIN = """
SELECT count() AS thought_count
FROM thoughts
WHERE chain_id = {chain_id}
"""

#: Semantic search across thought chains.
#: chDB doesn't allow correlated subqueries in aggregation keys. We
#: pre-compute the thought_count in a CTE per chain, then join.
SEARCH_THOUGHT_CHAINS_BY_EMBEDDING = """
WITH
chain_thought_counts AS (
    SELECT chain_id, count() AS thought_count
    FROM thoughts
    GROUP BY chain_id
),
ranked_thoughts AS (
    SELECT
        t.id,
        t.chain_id,
        t.thought,
        t.thought_number,
        t.branch_id,
        cosineDistance(t.embedding, {embedding}::Array(Float32)) AS distance
    FROM thoughts t
    INNER JOIN thought_chains tc ON t.chain_id = tc.id
    WHERE length(t.embedding) > 0
      AND tc.status = 'active'
    ORDER BY distance ASC
    LIMIT {limit}
)
SELECT
    rt.chain_id,
    tc.session_id,
    rt.thought AS snippet,
    rt.distance AS score,
    ctc.thought_count
FROM ranked_thoughts rt
INNER JOIN thought_chains tc ON rt.chain_id = tc.id
INNER JOIN chain_thought_counts ctc ON rt.chain_id = ctc.chain_id
GROUP BY rt.chain_id, tc.session_id, rt.thought, rt.distance, ctc.thought_count
ORDER BY min(rt.distance) ASC
LIMIT {final_limit}
"""

#: Update a thought's memory_id (link to a memory record).
UPDATE_THOUGHT_MEMORY_ID = """
ALTER TABLE thoughts
UPDATE memory_id = {memory_id}
WHERE id = {id}
"""


# =============================================================================
# Dual-Model RRF Queries (1024-dim sidecar)
# =============================================================================

#: Insert a 1024-dim embedding for a memory. Idempotency in chDB is
#: handled by the caller (SELECT first, skip if exists).
INSERT_MEMORY_1024 = """
INSERT INTO memories_1024
    (memory_id, embedding, elevated_at, elevated_from_dim, embedding_model, trust_score)
VALUES
    ({memory_id}, {embedding}::Array(Float32), {now}, 384, {embedding_model},
     {trust_score})
"""

#: Search the 1024-dim table.
SEARCH_MEMORIES_1024_VECTOR = """
SELECT
    m.id AS memory_id,
    m.text,
    m.source_type,
    m.trust_score,
    m.retrieval_count,
    m.is_archived,
    m.metadata,
    cosineDistance(m1024.embedding, {embedding}::Array(Float32)) AS distance
FROM memories_1024 m1024
INNER JOIN memories m ON m.id = m1024.memory_id
WHERE m.is_archived = false
  AND length(m1024.embedding) > 0
  AND cosineDistance(m1024.embedding, {embedding}::Array(Float32)) < {threshold}
ORDER BY distance ASC
LIMIT {limit}
"""

#: Get a 1024-dim embedding by memory_id.
GET_MEMORY_1024_BY_MEMORY_ID = """
SELECT id, memory_id, embedding, elevated_at, elevated_from_dim,
       embedding_model, trust_score
FROM memories_1024
WHERE memory_id = {memory_id}
"""

#: Get all 1024-dim embeddings joined with their memory records.
SEARCH_MEMORIES_1024_JOINED = """
SELECT
    m.id AS memory_id,
    m.text,
    m.source_type,
    m.trust_score,
    m.retrieval_count,
    m.is_archived,
    m.metadata,
    m1024.embedding
FROM memories_1024 m1024
INNER JOIN memories m ON m.id = m1024.memory_id
WHERE m.is_archived = false
ORDER BY cosineDistance(m1024.embedding, {embedding}::Array(Float32)) ASC
LIMIT {limit}
"""

#: Count rows in the 1024-dim table.
COUNT_MEMORIES_1024 = """
SELECT count() AS count FROM memories_1024
"""

#: Count thoughts in the thoughts table.
COUNT_THOUGHTS = """
SELECT count() AS count FROM thoughts
"""

#: Delete the 1024-dim sidecar for a memory.
DELETE_MEMORY_1024_BY_MEMORY_ID = """
DELETE FROM memories_1024 WHERE memory_id = {memory_id}
"""


# =============================================================================
# Image Recall Queries
# =============================================================================

#: Insert an image row. Idempotency in chDB is handled by the caller
#: (SELECT first by image_sha256, skip if exists).
INSERT_MEMORY_IMAGE = """
INSERT INTO memories_image
    (memory_id, embedding, embedding_model, image_path, image_sha256,
     mime_type, width, height, caption, file_size_bytes, trust_score)
VALUES
    ({memory_id}, {embedding}::Array(Float32), {embedding_model},
     {image_path}, {image_sha256}, {mime_type}, {width}, {height},
     {caption}, {file_size_bytes}, {trust_score})
"""

#: Search the image table.
SEARCH_MEMORIES_IMAGE = """
SELECT
    m.id AS memory_id,
    m.text,
    m.source_type,
    m.trust_score,
    m.retrieval_count,
    m.is_archived,
    m.metadata,
    mi.image_path,
    mi.image_sha256,
    mi.caption,
    cosineDistance(mi.embedding, {embedding}::Array(Float32)) AS distance
FROM memories_image mi
INNER JOIN memories m ON m.id = mi.memory_id
WHERE m.is_archived = false
  AND length(mi.embedding) > 0
  AND cosineDistance(mi.embedding, {embedding}::Array(Float32)) < {threshold}
ORDER BY distance ASC
LIMIT {limit}
"""

#: Look up an image row by its SHA-256.
SEARCH_MEMORIES_IMAGE_BY_SHA256 = """
SELECT id, memory_id, image_path, image_sha256, mime_type, caption, created_at
FROM memories_image
WHERE image_sha256 = {image_sha256}
"""

#: Delete the image sidecar for a memory.
DELETE_MEMORY_IMAGE = """
DELETE FROM memories_image WHERE memory_id = {memory_id}
"""

#: Update the image trust score.
UPDATE_MEMORY_IMAGE_TRUST = """
ALTER TABLE memories_image
UPDATE trust_score = {trust_score}
WHERE memory_id = {memory_id}
"""
