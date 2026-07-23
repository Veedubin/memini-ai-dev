-- Extend memories.source_type CHECK constraint to include 'github'
-- (superset of the existing constraint — existing rows still satisfy it)
--
-- Background: the GitHub triage poller (scripts/gh-triage-poller.py) polls
-- Veedubin repos for new issues/PRs and stores the wrapped text as a memory
-- with source_type='github'. The memories_source_type_check CHECK constraint
-- previously allowed only: session, file, web, boomerang, project, thought,
-- image. 'github' was rejected, so the poller's add_memory calls failed with
-- a constraint violation (swallowed as a non-fatal warning — zero github rows
-- existed in the live DB).
--
-- IMPORTANT: CREATE TABLE IF NOT EXISTS does NOT update CHECK constraints on
-- an existing live table. This migration must be applied manually to the live
-- DB. It is idempotent (DROP IF EXISTS + ADD) and non-destructive (additive —
-- the new constraint is a strict superset of the old one, so all existing
-- rows still satisfy it).
--
-- The application's _ensure_schema() also runs the equivalent
-- SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE constant at startup, so any new
-- deployment will get the 'github' value automatically. This migration is only
-- needed for EXISTING live databases that were created before this change.

ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought', 'image', 'github'));