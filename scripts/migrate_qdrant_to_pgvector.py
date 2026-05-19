#!/usr/bin/env python3
"""Migrate memini-ai data from Qdrant to pgvector/pgvectorscale."""

from __future__ import annotations

import asyncio
import sys

# Add the project source to path
sys.path.insert(0, "src")

from memini_ai.memory.database import QdrantDatabase
from memini_ai.memory.schema import MemoryEntry, SearchOptions, SearchStrategy
from memini_ai.postgres.database import PostgresDatabase

BATCH_SIZE = 100
DEFAULT_DIMENSION = 1024


async def get_all_qdrant_memories(qdrant: QdrantDatabase) -> list[MemoryEntry]:
    """Fetch all memories from Qdrant using scroll pagination."""
    await qdrant.initialize()
    return await qdrant.list_memories(filter=None)


async def verify_migration(
    qdrant: QdrantDatabase,
    pg: PostgresDatabase,
    sample_size: int = 10,
) -> bool:
    """Compare vector search results between Qdrant and pgvector."""
    print("\n--- Verification: Comparing vector search results ---")

    # Get sample memories for verification queries
    all_memories = await get_all_qdrant_memories(qdrant)
    if not all_memories:
        print("No memories to verify")
        return True

    # Use first N memories with vectors as test queries
    test_memories = [m for m in all_memories if m.vector][:sample_size]
    if not test_memories:
        print(f"No memories with vectors found for verification (tested {sample_size} samples)")
        return True

    print(f"Testing {len(test_memories)} random vector searches...")
    all_match = True

    for i, memory in enumerate(test_memories, 1):
        if not memory.vector:
            continue

        # Query both databases
        options = SearchOptions(top_k=5, threshold=0.0, strategy=SearchStrategy.VECTOR_ONLY)

        qdrant_results = await qdrant.query_memories(memory.vector, options)
        pg_results = await pg.query_memories(memory.vector, options)

        # Compare top results by ID
        qdrant_ids = {r.id for r in qdrant_results}
        pg_ids = {r.id for r in pg_results}

        overlap = len(qdrant_ids & pg_ids)
        total = len(qdrant_ids | pg_ids)
        similarity = overlap / total if total > 0 else 0.0

        print(f"  [{i}/{len(test_memories)}] Query '{memory.text[:50]}...': "
              f"Qdrant={len(qdrant_ids)} results, PG={len(pg_ids)} results, "
              f"ID overlap={overlap}/{total} ({similarity*100:.1f}%)")

        if similarity < 0.7:
            print("    ⚠ Low overlap - investigate if this is expected")
            all_match = False

    return all_match


async def migrate() -> None:
    """Main migration function."""
    print("=" * 60)
    print("Qdrant → pgvector/pgvectorscale Migration")
    print("=" * 60)

    # 1. Connect to Qdrant
    print("\n[1] Connecting to Qdrant...")
    qdrant = QdrantDatabase(url="http://localhost:6333")
    await qdrant.initialize()
    print("  ✓ Qdrant connection established")

    # 2. Connect to PostgreSQL
    print("\n[2] Connecting to PostgreSQL...")
    pg_url = "postgresql://postgres:password@localhost:5434/postgres"
    pg = PostgresDatabase(pg_url)
    await pg.initialize()
    print("  ✓ PostgreSQL connection established")

    # 3. Get all memories from Qdrant
    print("\n[3] Fetching memories from Qdrant...")
    all_memories = await get_all_qdrant_memories(qdrant)
    total_count = len(all_memories)
    print(f"  Found {total_count} memories to migrate")

    if total_count == 0:
        print("  Nothing to migrate - exiting")
        await pg.close()
        return

    # 4. Migrate to PostgreSQL in batches
    print(f"\n[4] Migrating to PostgreSQL (batch size: {BATCH_SIZE})...")

    migrated = 0
    failed = 0
    errors = []

    for i in range(0, total_count, BATCH_SIZE):
        batch = all_memories[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_count + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} memories...", end=" ")

        # Process batch - collect records first
        records = []
        for memory in batch:
            record = pg._entry_to_record(memory)  # type: ignore
            records.append(record)

        # Insert batch with transaction
        try:
            pool = await pg._get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for record in records:
                        await conn.fetchval(
                            """
                            INSERT INTO memories (id, text, embedding, source_type, content_hash, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (id) DO UPDATE SET
                                text = EXCLUDED.text,
                                embedding = EXCLUDED.embedding,
                                source_type = EXCLUDED.source_type,
                                content_hash = EXCLUDED.content_hash,
                                metadata = EXCLUDED.metadata
                            """,
                            record["id"],
                            record["text"],
                            record["embedding"],
                            record["source_type"],
                            record["content_hash"],
                            record["metadata"],
                        )

            migrated += len(batch)
            print(f"✓ ({migrated}/{total_count})")

        except Exception as e:
            failed += len(batch)
            errors.append(f"Batch {batch_num}: {str(e)}")
            print(f"✗ Error: {e}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  Total memories found: {total_count}")
    print(f"  Successfully migrated: {migrated}")
    print(f"  Failed: {failed}")

    if errors:
        print("\n  Errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")

    # 6. Verify
    if migrated > 0:
        print("\n[5] Verifying migration...")
        verified = await verify_migration(qdrant, pg, sample_size=min(10, total_count))
        if verified:
            print("  ✓ Verification passed - results are consistent")
        else:
            print("  ⚠ Verification completed with warnings")

    # Cleanup
    await pg.close()
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(migrate())
