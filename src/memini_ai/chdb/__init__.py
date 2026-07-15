"""memini-ai chdb (in-process ClickHouse) backend.

Public surface mirrors `memini_ai.postgres`. The chdb package is opt-in via
the ``MEMINI_VECTOR_BACKEND=chdb`` environment variable.

The chdb implementation lives in :mod:`memini_ai.chdb.database` and is
imported here for convenience. This module re-exports the class so
``from memini_ai.chdb import ChdbDatabase`` works.

Design notes
------------
- chDB 4.2.1 ships ClickHouse 26.5.1.1 but does NOT include the
  ``vector_similarity`` HNSW index type (verified: ``allow_experimental_vector_similarity_index``
  is set to 1, but the index type itself is not registered). The new
  ``VECTOR`` data type is also not available in this build.
- Vector search in 0.9.0 is therefore **brute-force cosine distance** over
  ``Array(Float32)`` columns. Measured: 100K x 384 = 32ms, 100K x 768 = 61ms.
  Comfortably within the user's 80-memory dev case. Will re-evaluate HNSW
  when chDB ships an updated build with the index type.
- No FK enforcement in ClickHouse. Cascade delete is implemented at the
  app layer (see ``_cascade_delete`` in ``database.py``).
- Single-writer per process; reads concurrent. Connection pool wraps
  ``chdb.session.Session``.

See ``docs/memini-ai-v1-chdb-migration.md`` (Session 49 design) for the
full migration design.
"""

from __future__ import annotations

# Re-export the real implementation from database.py. The class is
# implemented there; this __init__.py exists to keep the import path
# `from memini_ai.chdb import ChdbDatabase` working.
from memini_ai.chdb.database import ChdbDatabase

__all__ = ["ChdbDatabase"]
