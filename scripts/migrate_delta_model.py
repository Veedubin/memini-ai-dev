"""Migration script for Memory Delta Model (Schema v2) with Epoch Timestamp.

Adds columns for partial update support:
- supersedes_id: UUID references memories(id) - points to memory this partially updates
- structured_fields: JSONB - key-value fields for granular merge
- change_ratio: FLOAT DEFAULT 1.0 - fraction of content that is new/changed
- created_at_ms: BIGINT - Unix timestamp in milliseconds for temporal ordering

Also updates the memory_relationships table to include PARTIAL_UPDATE relationship type.
"""

from __future__ import annotations

import asyncio
import os


async def migrate() -> None:
    """Run the delta model migration."""
    import asyncpg

    db_url = os.environ.get("MEMINI_DB_URL", "")
    if not db_url:
        print("ERROR: MEMINI_DB_URL environment variable not set")
        return

    conn = await asyncpg.connect(db_url)

    try:
        print("Starting Memory Delta Model migration...")

        # Add supersedes_id column to memories table
        print("Adding supersedes_id column to memories...")
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS supersedes_id UUID REFERENCES memories(id) ON DELETE SET NULL
        """)
        print("  - supersedes_id column added")

        # Add structured_fields column to memories table
        print("Adding structured_fields column to memories...")
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS structured_fields JSONB DEFAULT NULL
        """)
        print("  - structured_fields column added")

        # Add change_ratio column to memories table
        print("Adding change_ratio column to memories...")
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS change_ratio FLOAT DEFAULT 1.0 CHECK (change_ratio >= 0 AND change_ratio <= 1)
        """)
        print("  - change_ratio column added")

        # Add created_at_ms column to memories table
        print("Adding created_at_ms column to memories...")
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS created_at_ms BIGINT
        """)
        print("  - created_at_ms column added")

        # Backfill existing rows with epoch ms from created_at
        print("Backfilling created_at_ms from created_at for existing rows...")
        await conn.execute("""
            UPDATE memories
            SET created_at_ms = GREATEST(
                EXTRACT(EPOCH FROM created_at) * 1000,
                EXTRACT(EPOCH FROM NOW()) * 1000
            )
            WHERE created_at_ms IS NULL
        """)
        print("  - Backfill complete")

        # Set NOT NULL constraint now that all rows have values
        print("Setting NOT NULL constraint on created_at_ms...")
        await conn.execute("""
            ALTER TABLE memories
            ALTER COLUMN created_at_ms SET NOT NULL
        """)
        print("  - NOT NULL constraint set")

        # Update memory_relationships CHECK constraint to include PARTIAL_UPDATE
        print("Updating memory_relationships relationship_type constraint...")
        await conn.execute("""
            ALTER TABLE memory_relationships DROP CONSTRAINT IF EXISTS memory_relationships_relationship_type_check;
            ALTER TABLE memory_relationships ADD CONSTRAINT memory_relationships_relationship_type_check
                CHECK (relationship_type IN ('SUPERSEDES', 'PARTIAL_UPDATE', 'RELATED_TO', 'CONTRADICTS', 'DERIVED_FROM'))
        """)
        print("  - PARTIAL_UPDATE relationship type added")

        # Create index for fast supersedes chain traversal
        print("Creating supersedes_id index...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_supersedes
            ON memories(supersedes_id) WHERE supersedes_id IS NOT NULL
        """)
        print("  - idx_memories_supersedes index created")

        # Create index for archived memories with supersedes
        print("Creating archived supersedes index...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_archived_supersedes
            ON memories(supersedes_id) WHERE is_archived = TRUE AND supersedes_id IS NOT NULL
        """)
        print("  - idx_memories_archived_supersedes index created")

        # Create index for created_at_ms (temporal ordering)
        print("Creating created_at_ms index...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at_ms
            ON memories(created_at_ms DESC)
        """)
        print("  - idx_memories_created_at_ms index created")

        print("\nMigration completed successfully!")
        print("\nNew schema features:")
        print("  - Memories can now track partial updates via supersedes_id")
        print("  - Structured fields (JSONB) enable granular field-level merging")
        print("  - change_ratio indicates what portion of memory is new/changed")
        print(
            "  - created_at_ms provides epoch milliseconds for temporal ordering and hierarchy"
        )
        print(
            "  - PARTIAL_UPDATE relationship type distinguishes delta updates from full replacements"
        )

    finally:
        await conn.close()


async def rollback() -> None:
    """Rollback the delta model migration."""
    import asyncpg

    db_url = os.environ.get("MEMINI_DB_URL", "")
    if not db_url:
        print("ERROR: MEMINI_DB_URL environment variable not set")
        return

    conn = await asyncpg.connect(db_url)

    try:
        print("Rolling back Memory Delta Model migration...")

        # Drop indexes
        await conn.execute("DROP INDEX IF EXISTS idx_memories_created_at_ms")
        await conn.execute("DROP INDEX IF EXISTS idx_memories_archived_supersedes")
        await conn.execute("DROP INDEX IF EXISTS idx_memories_supersedes")

        # Drop columns
        await conn.execute("ALTER TABLE memories DROP COLUMN IF EXISTS created_at_ms")
        await conn.execute("ALTER TABLE memories DROP COLUMN IF EXISTS change_ratio")
        await conn.execute(
            "ALTER TABLE memories DROP COLUMN IF EXISTS structured_fields"
        )
        await conn.execute("ALTER TABLE memories DROP COLUMN IF EXISTS supersedes_id")

        # Reset relationship_type constraint
        await conn.execute("""
            ALTER TABLE memory_relationships DROP CONSTRAINT IF EXISTS memory_relationships_relationship_type_check;
            ALTER TABLE memory_relationships ADD CONSTRAINT memory_relationships_relationship_type_check
                CHECK (relationship_type IN ('SUPERSEDES', 'RELATED_TO', 'CONTRADICTS', 'DERIVED_FROM'))
        """)

        print("Rollback completed successfully!")

    finally:
        await conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        asyncio.run(rollback())
    else:
        asyncio.run(migrate())
