# pgvector Migration Plan

> **Date**: 2026-05-18
> **Project**: Memini-ai v3.0
> **Goal**: Migrate from Qdrant to pgvector for unified database architecture

---

## Executive Summary

Memini-ai currently uses Qdrant for vector storage. This works for pure vector similarity search, but the architecture stores relational data (memory relationships, entity graphs, peer permissions) as JSON blobs inside Qdrant. This is a code smell.

**Proposed**: Migrate to pgvector (PostgreSQL extension for vectors). Keep Qdrant as optional fallback.

**Why now?**: The Trust Engine, Memory Graph, Multi-Peer, and Dialectic features all store relational data that would benefit from proper SQL joins, foreign keys, and aggregation queries.

---

## Why pgvector Over Qdrant?

### Current Pain Points

1. **Relationships as JSON**: Memory Graph stores SUPERSEDES/RELATED_TO/CONTRADICTS/DERIVED_FROM as JSON array in Qdrant payload. Can't query "find all CONTRADICTS relationships where target memory has trust_score < 0.3".

2. **Multi-Peer permissions app-enforced**: Permissions stored in JSON, enforced in Python code. Should be database-level.

3. **Trust decay loops**: Must fetch all memories, calculate decay in Python, write back. Should be: `UPDATE memories SET trust_score = trust_score * 0.95 WHERE last_accessed < NOW() - INTERVAL '7 days'`.

4. **Separate ops burden**: Qdrant is another server to run, backup, monitor. If you already run Postgres, pgvector is zero extra infrastructure.

5. **Can't join with user data**: User profiles stored separately. With pgvector, memories, users, peers all in same DB.

### Benchmarks

| Aspect | Qdrant | pgvector | Winner |
|--------|--------|----------|--------|
| Vector search speed | ~5ms | ~7ms | Qdrant |
| ANN accuracy | 95% | 93% | Qdrant |
| SQL joins | No | Yes | pgvector |
| Transactions | Limited | Full ACID | pgvector |
| Operational complexity | High | Low | pgvector |
| Ecosystem | Vector-only | Full Postgres | pgvector |

**Verdict**: pgvector is "good enough" for vector search (~95% performance), dramatically better for relational operations.

---

## Architecture Comparison

### Current Architecture (Qdrant)

```
┌─────────────┐     ┌─────────────┐
│   Memini-ai  │────▶│   Qdrant    │
│   (Python)   │     │ (Vectors +  │
└─────────────┘     │ JSON blobs) │
                    └─────────────┘
                         ▲
                         │ sync
                         │
                    ┌─────────────┐
                    │   Memory    │
                    │  relationships │
                    │  entities   │
                    │  (JSON in   │
                    │  Qdrant)    │
                    └─────────────┘
```

### Proposed Architecture (pgvector)

```
┌─────────────┐
│   Memini-ai  │────▶┌─────────────────────────────────────┐
│   (Python)   │     │          PostgreSQL                  │
└─────────────┘     │  ┌─────────┐  ┌──────────────────┐  │
                    │  │ vectors │  │ memories         │  │
                    │  └─────────┘  ├──────────────────┤  │
                    │  ┌─────────┐  │ memory_rels      │  │
                    │  │ entities │ ├──────────────────┤  │
                    │  └─────────┘  │ entities         │  │
                    │  ┌─────────┐  ├──────────────────┤  │
                    │  │ peers   │  │ entity_rels      │  │
                    │  └─────────┘  └──────────────────┘  │
                    └─────────────────────────────────────┘
```

---

## Schema Design

### Core Tables

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core memories with vector embedding
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    embedding vector(1024),  -- BGE-Large dimension
    source_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Trust Engine fields
    trust_score FLOAT DEFAULT 0.5 CHECK (trust_score >= 0 AND trust_score <= 1),
    retrieval_count INT DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE,
    last_accessed_at TIMESTAMP WITH TIME ZONE,

    -- Multi-peer
    peer_id UUID REFERENCES peers(id) ON DELETE SET NULL,

    -- Indexing
    content_hash VARCHAR(64),  -- SHA-256 for deduplication
    source_path TEXT,

    -- Metadata as JSONB for flexibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for vector similarity search
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);

-- Index for trust engine queries
CREATE INDEX idx_memories_trust ON memories(trust_score) WHERE NOT is_archived;
CREATE INDEX idx_memories_last_accessed ON memories(last_accessed_at) WHERE NOT is_archived;
```

### Relationships Table

```sql
-- Memory relationships (replaces JSON field)
CREATE TABLE memory_relationships (
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

-- Index for relationship queries
CREATE INDEX idx_mem_rel_source ON memory_relationships(source_id);
CREATE INDEX idx_mem_rel_target ON memory_relationships(target_id);
CREATE INDEX idx_mem_rel_type ON memory_relationships(relationship_type);
```

### Entities Table

```sql
-- Entities from knowledge graph extraction
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(50) NOT NULL CHECK (
        entity_type IN ('PERSON', 'ORGANIZATION', 'CONCEPT', 'CODE', 'PROJECT', 'LOCATION', 'UNKNOWN')
    ),
    canonical_name VARCHAR(500),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),

    -- Vector embedding for entity similarity
    embedding vector(1024),

    -- Ownership
    peer_id UUID REFERENCES peers(id) ON DELETE SET NULL,

    -- Occurrences
    mention_count INT DEFAULT 1,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(name, entity_type, peer_id)
);

-- Index for entity similarity
CREATE INDEX idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops);
```

### Entity Relationships

```sql
-- Entity relationships (from knowledge graph)
CREATE TABLE entity_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

CREATE INDEX idx_ent_rel_source ON entity_relationships(source_entity_id);
CREATE INDEX idx_ent_rel_target ON entity_relationships(target_entity_id);
```

### Peers Table

```sql
-- Peers for multi-user support
CREATE TABLE peers (
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

-- Memory sharing permissions
CREATE TABLE memory_sharing (
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
```

### User Profiles

```sql
-- User profiles (from user modeling)
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_id UUID UNIQUE REFERENCES peers(id) ON DELETE CASCADE,

    -- Profile data
    preferences JSONB DEFAULT '{}'::jsonb,
    communication_style VARCHAR(100) DEFAULT 'neutral',
    expertise_level VARCHAR(50) DEFAULT 'intermediate',

    -- Dialectic notes
    dialectic_notes JSONB DEFAULT '[]'::jsonb,

    -- Status
    warmed_up BOOLEAN DEFAULT FALSE,
    session_count INT DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Trust History (for auditing)

```sql
-- Trust score adjustments history
CREATE TABLE trust_adjustments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    old_score FLOAT NOT NULL,
    new_score FLOAT NOT NULL,
    signal VARCHAR(50) NOT NULL,  -- AGENT_USED, USER_CORRECTED, etc.
    adjustment_amount FLOAT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_trust_adj_memory ON trust_adjustments(memory_id);
CREATE INDEX idx_trust_adj_created ON trust_adjustments(created_at);
```

---

## Key SQL Queries (Post-Migration)

### Trust Decay (Cron Job)

```sql
-- Decay memories that haven't been accessed in 7 days
-- Formula: score * 0.5^(days*decay_rate/half_life)
UPDATE memories
SET trust_score = trust_score * POWER(0.5, (
    EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 86400.0 * decay_rate / 90
  ))
WHERE
    is_archived = FALSE
    AND trust_score > 0.1
    AND last_accessed_at < NOW() - INTERVAL '7 days'
    AND decay_rate IS NOT NULL;
```

### Find Contradictions with Low Trust

```sql
-- Find contradicting memories where both have low trust
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
WHERE
    r.relationship_type = 'CONTRADICTS'
    AND m1.is_archived = FALSE
    AND m2.is_archived = FALSE
    AND m1.trust_score < 0.5
    AND m2.trust_score < 0.5
ORDER BY (m1.trust_score + m2.trust_score) ASC;
```

### Archive Low-Trust Memories

```sql
-- Archive memories below threshold
UPDATE memories
SET is_archived = TRUE, updated_at = NOW()
WHERE trust_score < 0.2 AND is_archived = FALSE;
```

### Complex Multi-Peer Query

```sql
-- Find memories shared with a specific peer that relate to a query
SELECT DISTINCT m.*
FROM memories m
JOIN memory_sharing ms ON m.id = ms.memory_id
JOIN memory_relationships mr ON m.id = mr.source_id
WHERE
    ms.peer_id = 'target-peer-uuid'
    AND ms.permission IN ('SHARED', 'INHERITED')
    AND mr.relationship_type = 'RELATED_TO'
    AND m.is_archived = FALSE
ORDER BY m.retrieval_count DESC
LIMIT 20;
```

### Entity Inference (Transitive Closure)

```sql
-- Find all entities connected to a source entity within N hops
WITH RECURSIVE entity_chain AS (
    -- Base case: direct relationships
    SELECT
        source_entity_id,
        target_entity_id,
        1 AS depth,
        ARRAY[source_entity_id, target_entity_id] AS path
    FROM entity_relationships

    UNION ALL

    -- Recursive case: extend path
    SELECT
        er.source_entity_id,
        er.target_entity_id,
        ec.depth + 1,
        ec.path || er.target_entity_id
    FROM entity_relationships er
    JOIN entity_chain ec ON er.source_entity_id = ec.target_entity_id
    WHERE
        er.target_entity_id != ALL(ec.path)
        AND ec.depth < 3  -- Max inference depth
)
SELECT DISTINCT e.*, ec.depth, ec.path
FROM entity_chain ec
JOIN entities e ON ec.target_entity_id = e.id
WHERE ec.source_entity_id = 'source-entity-uuid'
ORDER BY ec.depth;
```

---

## Implementation Plan

### Phase 1: Abstraction Layer
- [ ] Add `VectorDatabase` abstract base class in `database.py`
- [ ] Rename current `MemoryDatabase` to `QdrantDatabase`
- [ ] Add `PostgresDatabase` stub with same interface
- [ ] Update config to select database backend

### Phase 2: pgvector Implementation
- [ ] Create `src/memini_ai/postgres/` module
- [ ] Implement `PostgresDatabase` with all required methods
- [ ] Create SQL schema in `postgres/schema.sql`
- [ ] Implement vector search using `embedding <=> query` operator

### Phase 3: Migration Script
- [ ] Create `scripts/migrate_qdrant_to_pgvector.py`
- [ ] Export memories from Qdrant
- [ ] Transform and import to PostgreSQL
- [ ] Verify vector similarity results match

### Phase 4: Feature Updates
- [ ] Update Trust Engine to use SQL aggregation
- [ ] Update Memory Graph to use relationship tables
- [ ] Update Multi-Peer to use foreign keys
- [ ] Update Knowledge Graph to use entity tables

### Phase 5: Testing & Deployment
- [ ] Add 50+ pgvector-specific tests
- [ ] Performance benchmarking
- [ ] Update documentation
- [ ] Deprecation notice for Qdrant-only mode

---

## Abstraction Interface

```python
# src/memini_ai/database.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
import numpy as np

@dataclass
class MemoryMatch:
    memory_id: str
    score: float
    memory: "MemoryEntry"

class VectorDatabase(ABC):
    """Abstract interface for vector database backends."""

    @abstractmethod
    async def search_vectors(
        self,
        query: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None
    ) -> list[MemoryMatch]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def insert_memory(self, memory: "MemoryEntry") -> str:
        """Insert a memory entry."""
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> "MemoryEntry | None":
        """Get a memory by ID."""
        pass

    @abstractmethod
    async def update_memory(self, memory: "MemoryEntry") -> None:
        """Update a memory entry."""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: "RelationshipType",
        confidence: float = 1.0
    ) -> None:
        """Upsert a memory relationship."""
        pass

    @abstractmethod
    async def get_relationships(
        self,
        memory_id: str,
        rel_type: "RelationshipType | None" = None
    ) -> list["Relationship"]:
        """Get relationships for a memory."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        pass

# Factory function
def create_database(config: "MeminiConfig") -> VectorDatabase:
    """Create database instance based on config."""
    if config.database_url:
        from .postgres import PostgresDatabase
        return PostgresDatabase(config.database_url)
    else:
        from .qdrant import QdrantDatabase
        return QdrantDatabase(config.qdrant_url)
```

---

## File Structure

```
src/memini_ai/
├── database.py              # REFACTORED: Abstract base class + factory
├── qdrant.py               # RENAME: Current database.py → qdrant.py
├── postgres/               # NEW: pgvector module
│   ├── __init__.py
│   ├── database.py         # PostgresDatabase implementation
│   ├── schema.py           # SQL schema definitions
│   ├── queries.py          # Reusable SQL query builders
│   └── migrations/         # Alembic or manual migrations
│       └── 001_initial.sql
├── memory/
│   ├── database.py         # ADAPTER: Wraps VectorDatabase
│   └── system.py           # Uses VectorDatabase via adapter
└── server.py               # UPDATED: Database selection
```

---

## Dependencies

```toml
# pyproject.toml additions
dependencies = [
    # ... existing ...
    "asyncpg>=0.29.0",  # Async PostgreSQL driver
]

[dependency-groups]
dev = [
    # ... existing ...
    "pytest-asyncio>=0.23",
]
```

---

## Rollback Plan

If pgvector migration fails:
1. Keep `QdrantDatabase` as default
2. Set `MEMINI_DB_URL=""` or unset to use Qdrant
3. No data loss - migration script can be re-run

---

## Success Criteria

1. **Functional**: All existing features work with pgvector
2. **Compatible**: Vector search results within 5% of Qdrant
3. **Tested**: 50+ new tests for pgvector-specific queries
4. **Documented**: Migration guide in docs/
5. **Reversible**: Can switch back to Qdrant via config

---

## Timeline Estimate

- Phase 1 (Abstraction): 1 day
- Phase 2 (Implementation): 3-4 days
- Phase 3 (Migration): 1-2 days
- Phase 4 (Feature Updates): 2-3 days
- Phase 5 (Testing): 1-2 days

**Total**: ~8-12 days

---

*Document created: 2026-05-18*
*Next review: Before starting Phase 5 work*
