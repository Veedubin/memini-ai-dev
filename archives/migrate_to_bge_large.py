#!/usr/bin/env python3
"""
Migrate memini-postgres memories from BGE-M3 (1024-dim) to BGE-Large (1024-dim).

Non-destructive: populates the `embedding_bge_large` column alongside the existing
BGE-M3 and MiniLM columns. The `embedding_model` field stays as-is (it tracks
the PRIMARY model, not all models).

Run:
    python3 migrate_to_bge_large.py [--dry-run] [--batch 10] [--db-url <url>]

Safety:
- Backs up each row's id + text to a JSONL file before any write
- Atomic: writes happen in transactions, rolled back on error
- Resumable: skips rows that already have embedding_bge_large set
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

# Defaults — override via --db-url or env var
DEFAULT_DB_URL = "postgresql://postgres:password@localhost:5434/postgres"

# BGE-Large spec
BGE_LARGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_LARGE_DIM = 1024

# Where to save the pre-migration backup
DEFAULT_BACKUP_PATH = "/home/jcharles/Projects/MCP-Servers/archives/memini-embedding-migration-2026-07-10/memini-migration-to-bge-large-backup.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migrate memini-postgres memories to BGE-Large (populate embedding_bge_large column)"
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
    return p.parse_args()


def load_bge_large(device: str):
    """Load BGE-Large model. Returns (model, device_resolved)."""
    from sentence_transformers import SentenceTransformer

    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    log(f"loading BGE-Large on {resolved_device}...")
    t0 = time.time()
    model = SentenceTransformer(BGE_LARGE_MODEL, device=resolved_device)
    elapsed = time.time() - t0
    log(f"BGE-Large loaded in {elapsed:.1f}s")
    return model, resolved_device


def fetch_memories(conn, limit: Optional[int]):
    """Fetch all non-archived memories that need BGE-Large embedding (don't have embedding_bge_large yet)."""
    query = """
        SELECT id, text
        FROM memories
        WHERE is_archived = false
          AND embedding_bge_large IS NULL
        ORDER BY created_at ASC
    """
    if limit:
        query += " LIMIT %s"
    with conn.cursor() as cur:
        cur.execute(query, [limit] if limit else [])
        return cur.fetchall()


def write_backup(backup_path: str, rows: list) -> None:
    """Write the pre-migration backup as JSONL."""
    Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w") as f:
        for row in rows:
            mem_id, text = row
            f.write(json.dumps({"id": str(mem_id), "text": text}) + "\n")
    log(f"backup written to {backup_path} ({len(rows)} rows)")


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with BGE-Large."""
    import numpy as np

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=len(texts),
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def update_memories(conn, updates: list[tuple[str, list[float]]]) -> None:
    """Update embedding_bge_large for a batch of memories."""
    from psycopg2.extras import execute_values

    with conn.cursor() as cur:
        execute_values(
            cur,
            "UPDATE memories SET embedding_bge_large = v.embedding::vector FROM (VALUES %s) AS v(id, embedding) WHERE memories.id = v.id::uuid",
            [(mem_id, f"[{','.join(str(x) for x in vec)}]") for mem_id, vec in updates],
        )
    conn.commit()


def main() -> int:
    args = parse_args()

    log(f"connecting to {args.db_url[:60]}...")
    conn = psycopg2.connect(args.db_url)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'embedding_bge_large'
                """
            )
            if cur.fetchone() is None:
                log("ERROR: column embedding_bge_large does not exist")
                log("Run migration 000006 first")
                return 1
            log("column embedding_bge_large exists ✓")

        log("fetching memories to migrate to BGE-Large...")
        rows = fetch_memories(conn, args.limit)
        log(f"found {len(rows)} memories to migrate (need BGE-Large vector)")

        if not rows:
            log("nothing to do. all memories already have embedding_bge_large")
            return 0

        if args.dry_run:
            log("[DRY-RUN] would migrate these memories:")
            for i, (mem_id, text) in enumerate(rows[:10]):
                log(f"  {i + 1}. {mem_id} — {text[:80]}...")
            if len(rows) > 10:
                log(f"  ... and {len(rows) - 10} more")
            log("[DRY-RUN] no changes made")
            return 0

        write_backup(args.backup_path, rows)

        model, device = load_bge_large(args.device)
        log(f"using device: {device}")

        total = len(rows)
        processed = 0
        errors: list[str] = []
        start_time = time.time()

        for batch_start in range(0, total, args.batch):
            batch = rows[batch_start : batch_start + args.batch]
            batch_ids = [str(r[0]) for r in batch]
            batch_texts = [r[1] for r in batch]

            try:
                vectors = embed_batch(model, batch_texts)
                for v in vectors:
                    assert len(v) == BGE_LARGE_DIM
                update_memories(conn, list(zip(batch_ids, vectors)))
                processed += len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                log(
                    f"  ✓ {processed}/{total} memories got BGE-Large "
                    f"({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)"
                )
            except Exception as e:
                log(f"  ✗ batch failed at offset {batch_start}: {e}")
                errors.append(f"batch {batch_start}: {e}")
                conn.rollback()
                continue

        total_elapsed = time.time() - start_time
        log("")
        log("=" * 60)
        log("BGE-LARGE MIGRATION COMPLETE")
        log(f"  Total memories:     {total}")
        log(f"  Migrated:           {processed}")
        log(f"  Errors:             {len(errors)}")
        log(f"  Elapsed:            {total_elapsed:.1f}s")
        log(f"  Column populated:   embedding_bge_large (1024-dim)")
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
