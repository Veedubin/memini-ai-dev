# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-18

### Features

- pgvector/pgvectorscale backend with StreamingDiskANN index
- VectorDatabase ABC for database abstraction
- PostgresDatabase class with asyncpg support
- New `postgres/` module with schema and queries
- Migration script: `scripts/migrate_qdrant_to_pgvector.py`
- New config options: `MEMINI_DB_URL`, `db_pool_size`, `db_min_size`, `db_max_size`

### Tests

- 38 new tests for PostgresDatabase

### Bug Fixes

- N/A

### Breaking Changes

- None (backward compatible with Qdrant)