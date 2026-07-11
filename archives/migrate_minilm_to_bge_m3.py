#!/usr/bin/env python3
"""
Migrate memini-postgres memories from MiniLM-L6-v2 (384-dim) to BGE-M3 (1024-dim).

Non-destructive: adds a new column `embedding_bge_m3 vector(1024)` alongside the
existing 384-dim `embedding` column. Original 384-dim vectors are preserved.

Run:
    python3 migrate_minilm_to_bge_m3.py [--dry-run] [--batch 10] [--db-url <url>]

Safety:
- Backs up each row's text + old vector to a JSON file before any write
- Atomic: writes happen in transactions, rolled back on error
- Resumable: skips rows that already have embedding_bge_m3 set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values

# Defaults — override via --db-url or env var
DEFAULT_DB_URL = "postgresql://postgres:password@localhost:5434/postgres"

# BGE-M3 spec
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIM = 1024

# Where to save the pre-migration backup (each row's id, text, old vector)
DEFAULT_BACKUP_PATH = "/tmp/opencode/memini-migration-backup.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migrate memini-postgres memories from MiniLM (384-dim) to BGE-M3 (1024-dim)"
    )
    p.add_argument(
        "--db-url",
        default=os.environ.get("MEMINI_DB_URL", DEFAULT_DB_URL),
        help="PostgreSQL connection string (default: $MEMINI_DB_URL or memini-postgres local)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Memories per batch (default: 10)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without making changes",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=os.environ.get("NEURALGENTICS_EMBED_DEVICE", "auto"),
        help="Embedding model device (default: auto — uses cuda if available)",
    )
    p.add_argument(
        "--backup-path",
        default=DEFAULT_BACKUP_PATH,
        help=f"Where to write the per-row backup (default: {DEFAULT_BACKUP_PATH})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N memories (for testing)",
    )
    p.add_argument(
        "--source-types",
        default=None,
        help="Comma-separated list of source_types to migrate (default: all)",
    )
    return p.parse_args()


def load_bge_m3(device: str):
    """Load BGE-M3 model. Returns (model, device_resolved)."""
    from sentence_transformers import SentenceTransformer

    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    log(f"loading BGE-M3 on {resolved_device}...")
    t0 = time.time()
    model = SentenceTransformer(BGE_M3_MODEL, device=resolved_device)
    elapsed = time.time() - t0
    log(f"BGE-M3 loaded in {elapsed:.1f}s (dim={model.get_embedding_dimension()})")
    return model, resolved_device


def fetch_memories(conn, source_types: Optional[list[str]], limit: Optional[int]):
    """Fetch all non-archived memories that need migration (don't have BGE-M3 vector yet)."""
    query = """
        SELECT id, text, embedding
        FROM memories
        WHERE is_archived = false
          AND embedding_bge_m3 IS NULL
        ORDER BY created_at ASC
    """
    params: list = []
    if source_types:
        query = query.replace(
            "WHERE is_archived = false",
            "WHERE is_archived = false AND source_type = ANY(%s)",
            1,
        )
        params.append(source_types)
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def write_backup(backup_path: str, rows: list) -> None:
    """Write the pre-migration backup as JSONL (one row per line)."""
    Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w") as f:
        for row in rows:
            mem_id, text, embedding = row
            # Convert embedding to list for JSON serialization
            embedding_list = list(embedding) if embedding is not None else None
            f.write(
                json.dumps(
                    {
                        "id": str(mem_id),
                        "text": text,
                        "old_embedding_384": embedding_list,
                    }
                )
                + "\n"
            )
    log(f"backup written to {backup_path} ({len(rows)} rows)")


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with BGE-M3."""
    import numpy as np

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=len(texts),
        normalize_embeddings=True,  # BGE-M3 recommends normalized vectors
    )
    return [v.tolist() for v in vectors]


def update_memories(conn, updates: list[tuple[str, list[float]]]) -> None:
    """Update embedding_bge_m3 for a batch of memories."""
    with conn.cursor() as cur:
        # Use execute_values for efficient batch update
        execute_values(
            cur,
            "UPDATE memories SET embedding_bge_m3 = v.embedding::vector FROM (VALUES %s) AS v(id, embedding) WHERE memories.id = v.id::uuid",
            [(mem_id, f"[{','.join(str(x) for x in vec)}]") for mem_id, vec in updates],
        )
    conn.commit()


def main() -> int:
    args = parse_args()

    log(f"connecting to {args.db_url[:60]}...")
    conn = psycopg2.connect(args.db_url)
    conn.autocommit = False

    try:
        # Check if the new column exists
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'embedding_bge_m3'
                """
            )
            if cur.fetchone() is None:
                log("ERROR: column embedding_bge_m3 does not exist")
                log(
                    "Run: ALTER TABLE memories ADD COLUMN embedding_bge_m3 vector(1024);"
                )
                return 1
            log("column embedding_bge_m3 exists ✓")

        # Fetch memories to migrate
        source_types = args.source_types.split(",") if args.source_types else None
        log(f"fetching memories to migrate (source_types={source_types or 'all'})...")
        rows = fetch_memories(conn, source_types, args.limit)
        log(f"found {len(rows)} memories to migrate")

        if not rows:
            log("nothing to do. all memories already have embedding_bge_m3")
            return 0

        if args.dry_run:
            log("[DRY-RUN] would migrate these memories:")
            for i, (mem_id, text, _) in enumerate(rows[:10]):
                log(f"  {i + 1}. {mem_id} — {text[:80]}...")
            if len(rows) > 10:
                log(f"  ... and {len(rows) - 10} more")
            log("[DRY-RUN] no changes made")
            return 0

        # Backup before any writes
        write_backup(args.backup_path, rows)

        # Load the model
        model, device = load_bge_m3(args.device)
        log(f"using device: {device}")

        # Process in batches
        total = len(rows)
        processed = 0
        errors: list[str] = []
        start_time = time.time()

        for batch_start in range(0, total, args.batch):
            batch = rows[batch_start : batch_start + args.batch]
            batch_ids = [str(r[0]) for r in batch]
            batch_texts = [r[1] for r in batch]

            try:
                # Embed
                vectors = embed_batch(model, batch_texts)

                # Verify dimensions
                for v in vectors:
                    assert len(v) == BGE_M3_DIM, (
                        f"expected {BGE_M3_DIM}-dim vector, got {len(v)}"
                    )

                # Update
                update_memories(conn, list(zip(batch_ids, vectors)))
                processed += len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                log(
                    f"  ✓ {processed}/{total} memories migrated "
                    f"({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)"
                )
            except Exception as e:
                log(f"  ✗ batch failed at offset {batch_start}: {e}")
                errors.append(f"batch {batch_start}: {e}")
                # Roll back this batch's transaction (autocommit=False already)
                conn.rollback()
                continue

        total_elapsed = time.time() - start_time
        log("")
        log("=" * 60)
        log("MIGRATION COMPLETE")
        log(f"  Total memories:     {total}")
        log(f"  Migrated:           {processed}")
        log(f"  Errors:             {len(errors)}")
        log(f"  Elapsed:            {total_elapsed:.1f}s")
        log(f"  From → To:          MiniLM-L6-v2 (384-dim) → BGE-M3 (1024-dim)")
        log(f"  Backup file:        {args.backup_path}")
        log("=" * 60)
        if errors:
            log("First errors:")
            for e in errors[:5]:
                log(f"  - {e}")
            return 1
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
