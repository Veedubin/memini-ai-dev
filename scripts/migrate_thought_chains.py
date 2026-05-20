#!/usr/bin/env python3
"""Idempotent migration script for thought_chains and thoughts tables.

Creates the thought_chains and thoughts tables with all indexes
and updates the memories source_type CHECK constraint to include 'thought'.

Run: python scripts/migrate_thought_chains.py
"""

import asyncio
import os
import sys

import asyncpg

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memini_ai.config import get_config
from memini_ai.postgres.schema import (
    SQL_CREATE_THOUGHT_CHAINS_INDEXES,
    SQL_CREATE_THOUGHT_CHAINS_TABLE,
    SQL_CREATE_THOUGHTS_INDEXES,
    SQL_CREATE_THOUGHTS_TABLE,
    SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK,
)


async def migrate() -> None:
    """Run the thought chains migration."""
    config = get_config()
    db_url = config.db_url

    if not db_url:
        print(
            "ERROR: MEMINI_DB_URL not set. Set it in environment or .opencode/memini-ai/config.json"
        )
        sys.exit(1)

    print(f"Connecting to PostgreSQL at {db_url.split('@')[-1]}...")
    conn = await asyncpg.connect(db_url)

    try:
        # Create thought_chains table
        print("Creating thought_chains table...")
        await conn.execute(SQL_CREATE_THOUGHT_CHAINS_TABLE)

        # Create thought_chains indexes
        print("Creating thought_chains indexes...")
        await conn.execute(SQL_CREATE_THOUGHT_CHAINS_INDEXES)

        # Create thoughts table
        print("Creating thoughts table...")
        await conn.execute(SQL_CREATE_THOUGHTS_TABLE)

        # Create thoughts indexes
        print("Creating thoughts indexes...")
        await conn.execute(SQL_CREATE_THOUGHTS_INDEXES)

        # Update memories source_type CHECK constraint
        print("Updating memories source_type CHECK constraint...")
        await conn.execute(SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK)

        # Verify
        tc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'thought_chains'"
        )
        t_count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'thoughts'"
        )

        print("\nMigration complete!")
        print(f"  thought_chains table: {'EXISTS' if tc_count else 'MISSING'}")
        print(f"  thoughts table: {'EXISTS' if t_count else 'MISSING'}")
        print("  memories source_type: includes 'thought'")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
