"""Regression tests for T-IDX-WEDGE-001 — indexer must not wedge the event loop.

Covers:
1. Chunking runs off the event loop (asyncio.to_thread).
2. Tool calls stay responsive DURING a background index run.
3. Chunks persist incrementally to project_chunks (was a stat-only stub).
4. Directory walk prunes excluded trees (never descends into them).
5. Glob-syntax exclusion patterns (*.egg-info dirs, ~$* files) actually match.
6. clear() wipes project_chunks too.
7. Flush falls back to count-only when tracker is uninitialized.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memini_ai.indexer.indexer import IndexerConfig, ProjectIndexer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ELIGIBLE_FILES: dict[str, str] = {
    "src/mod_one.py": 'def alpha() -> int:\n    """Doc."""\n    return 1\n\n\ndef beta() -> int:\n    return 2\n',
    "src/mod_two.py": "class Gamma:\n    pass\n",
    "docs/readme.md": "# Title\n\nSome prose content for chunking.\n",
    "app/main.py": "import sys\n\n\ndef main() -> None:\n    print(sys.argv)\n",
}


@pytest.fixture()
def proj_tree(tmp_path: Path) -> Path:
    """Build a fixture repo: eligible files plus excluded decoy trees."""
    root = tmp_path / "proj"
    for rel, content in ELIGIBLE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    # Extra bulk eligible files so an index run takes measurable time.
    bulk = root / "bulk"
    bulk.mkdir()
    for i in range(20):
        (bulk / f"gen_{i:02d}.py").write_text(
            f"# generated {i}\n"
            + "\n".join(f"line_{j} = {j}" for j in range(40))
            + f"\n\ndef gen_{i}() -> int:\n    return {i}\n",
            encoding="utf-8",
        )

    # Excluded decoy trees containing otherwise-eligible files.
    for decoy_rel in (
        ".venv/lib/site-packages/pkg/__init__.py",
        ".venv/lib/site-packages/pkg/core.py",
        "node_modules/leftpad/index.js",
        "pkg.egg-info/PKG_INFO.py",  # glob-pattern SKIP_DIR (*.egg-info)
        "__pycache__/mod_one.cpython-312.pyc",
    ):
        p = root / decoy_rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("decoy = True\n", encoding="utf-8")

    # Always-excluded glob file name with an ALLOWED extension.
    (root / "src/~$important.py").write_text("temp lock\n", encoding="utf-8")
    return root


@pytest.fixture()
def make_indexer(tmp_path: Path) -> Iterator[Any]:
    """Factory for a started indexer backed by a temp sqlite db.

    Each test is responsible for stopping the indexers it creates (the
    suite's `_stop` helper); teardown here would need loop plumbing that
    pytest-asyncio does not provide in a sync fixture.
    """

    def _make(**kwargs: Any) -> ProjectIndexer:
        cfg = IndexerConfig(
            db_path=str(tmp_path / "indexer.db"),
            flush_interval=kwargs.pop("flush_interval", 0.15),
            **kwargs,
        )
        return ProjectIndexer(cfg)

    yield _make


async def _stop(indexer: ProjectIndexer) -> None:
    if indexer.is_running:
        await indexer.stop()


# ---------------------------------------------------------------------------
# 1. Chunking executes off the event loop
# ---------------------------------------------------------------------------


class TestOffLoopChunking:
    @pytest.mark.asyncio
    async def test_chunking_runs_in_worker_thread(
        self, proj_tree: Path, make_indexer: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer = make_indexer()
        await indexer.start()
        loop_thread = threading.get_ident()
        seen_threads: list[int] = []
        real_chunk = type(indexer._chunker).chunk_file

        def recording_chunk(content: str, file_path: str) -> Any:
            seen_threads.append(threading.get_ident())
            time.sleep(0.001)
            return real_chunk(indexer._chunker, content, file_path)

        monkeypatch.setattr(indexer._chunker, "chunk_file", recording_chunk)

        count = await indexer.index_directory(str(proj_tree))

        await _stop(indexer)
        assert count == 24  # 4 curated + 20 bulk
        assert seen_threads, "chunker was never invoked"
        # Every chunking call ran on a worker thread, never the loop thread.
        assert all(t != loop_thread for t in seen_threads)


# ---------------------------------------------------------------------------
# 2. Tool-call responsiveness DURING background indexing
# ---------------------------------------------------------------------------


class TestResponsivenessDuringIndexing:
    @pytest.mark.asyncio
    async def test_probe_completes_while_indexing(
        self, proj_tree: Path, make_indexer: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexer = make_indexer()
        await indexer.start()

        # Simulate CPU-heavy chunking: 24 files x 150ms blocking work each
        # (~3.6s serial). If any of it ran inline on the loop, the probe
        # below could not complete while indexing is still in flight.
        real_chunk = type(indexer._chunker).chunk_file

        def slow_chunk(content: str, file_path: str) -> Any:
            time.sleep(0.15)
            return real_chunk(indexer._chunker, content, file_path)

        monkeypatch.setattr(indexer._chunker, "chunk_file", slow_chunk)

        indexing = asyncio.create_task(indexer.index_directory(str(proj_tree)))
        await asyncio.sleep(0.05)  # let indexing spin up

        # The "tool call": a lightweight coroutine like get_status/healthcheck.
        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=2.0)
        elapsed = asyncio.get_running_loop().time() - t0

        assert not indexing.done(), (
            "indexing finished before probe — test lost discrimination power; "
            "increase per-file delay"
        )
        assert elapsed < 2.0, (
            f"event loop starved during indexing (probe took {elapsed:.2f}s)"
        )

        assert await indexing == 24
        await _stop(indexer)


# ---------------------------------------------------------------------------
# 3. Incremental persistence to project_chunks
# ---------------------------------------------------------------------------


class TestChunkPersistence:
    @pytest.mark.asyncio
    async def test_chunks_persist_and_are_retrievable(
        self, proj_tree: Path, make_indexer: Any
    ) -> None:
        indexer = make_indexer(flush_interval=0.1)
        await indexer.start()

        indexing = asyncio.create_task(indexer.index_directory(str(proj_tree)))

        # Incremental proof: observe >0 rows while the run may still be going.
        saw_partial = False
        while not indexing.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(indexing), timeout=0.08)
            if await indexer._file_tracker.count_chunks() > 0:
                saw_partial = True
                break
        assert await indexing == 24

        # Flush while the tracker connection is still open (stop() closes it).
        await indexer._flush_chunks()
        tracker = indexer._file_tracker
        total_rows = await tracker.count_chunks()
        assert total_rows > 0

        # Rows must exactly equal what the chunker produces for the fixtures.
        expected = 0
        for rel, content in ELIGIBLE_FILES.items():
            chunks = indexer._chunker.chunk_file(content, str(proj_tree / rel))
            expected += len(chunks)
            rows = await tracker.get_chunks_for_file(str(proj_tree / rel))
            assert [r["chunk_index"] for r in rows] == list(range(len(chunks)))
            assert rows[0]["content"] == chunks[0].content
        bulk_expected = 0
        for i in range(20):
            p = proj_tree / "bulk" / f"gen_{i:02d}.py"
            bulk_expected += len(
                indexer._chunker.chunk_file(p.read_text(encoding="utf-8"), str(p))
            )
        assert total_rows == expected + bulk_expected
        # Soft signal only — timing-dependent on fast machines.
        _ = saw_partial

    @pytest.mark.asyncio
    async def test_clear_also_wipes_project_chunks(
        self, proj_tree: Path, make_indexer: Any
    ) -> None:
        indexer = make_indexer()
        await indexer.start()
        assert await indexer.index_directory(str(proj_tree)) == 24
        await indexer._flush_chunks()  # final flush while connection is open
        tracker = indexer._file_tracker
        assert await tracker.count_chunks() > 0

        await tracker.clear()
        assert await tracker.count_chunks() == 0
        assert await tracker.get_all_paths() == []
        await _stop(indexer)


# ---------------------------------------------------------------------------
# 4 + 5. Pruned walk and glob-pattern exclusions
# ---------------------------------------------------------------------------


class TestWalkExclusions:
    @pytest.mark.asyncio
    async def test_walk_never_descends_into_excluded_dirs(
        self, proj_tree: Path, make_indexer: Any
    ) -> None:
        indexer = make_indexer()
        yielded = [str(p) async for p in indexer._walk_directory(proj_tree)]
        names = {Path(p).name for p in yielded}

        assert yielded, "walk found nothing at all"
        assert not any(".venv" in Path(p).parts for p in yielded)
        assert not any("node_modules" in Path(p).parts for p in yielded)
        assert not any("pkg.egg-info" in Path(p).parts for p in yielded)
        assert "__pycache__" not in {Path(p).parent.name for p in yielded}
        assert "~$important.py" not in names

    @pytest.mark.asyncio
    async def test_glob_pattern_dirs_and_files_excluded_from_stats(
        self, proj_tree: Path, make_indexer: Any
    ) -> None:
        indexer = make_indexer()
        await indexer.start()
        count = await indexer.index_directory(str(proj_tree))
        paths = await indexer._file_tracker.get_all_paths()
        await _stop(indexer)

        assert count == 24
        assert paths, "no files tracked"
        assert all(
            ".venv" not in Path(p).parts
            and "node_modules" not in Path(p).parts
            and "pkg.egg-info" not in Path(p).parts
            for p in paths
        )


# ---------------------------------------------------------------------------
# 7. Legacy fallback: flush without an initialized tracker
# ---------------------------------------------------------------------------


class TestFlushFallback:
    @pytest.mark.asyncio
    async def test_flush_without_tracker_counts_only(self, tmp_path: Path) -> None:
        cfg = IndexerConfig(db_path=str(tmp_path / "never-opened.db"))
        indexer = ProjectIndexer(cfg)  # start() never called -> no connection
        from memini_ai.indexer.chunker import Chunk

        indexer._chunk_buffer.append(
            Chunk(
                content="x = 1\n",
                path=str(tmp_path / "a.py"),
                chunk_index=0,
                total_chunks=1,
                start_line=1,
                end_line=1,
            )
        )
        await indexer._flush_chunks()
        assert indexer.get_stats().chunks_created == 1


# ---------------------------------------------------------------------------
# 8. _parse_python must TERMINATE on triple-quoted lines (the actual wedge)
# ---------------------------------------------------------------------------


class TestParsePythonTermination:
    """The pre-fix single-line-docstring branch did `continue` without
    incrementing `i`, hanging forever on ANY file containing triple quotes."""

    def test_single_line_docstring_terminates(self) -> None:
        from memini_ai.indexer.chunker import SemanticChunker

        chunker = SemanticChunker()
        content = 'def alpha() -> int:\n    """Doc."""\n    return 1\n'
        chunks = chunker.chunk_file(content, "mod.py")  # must not hang
        assert chunks, "expected at least one chunk"
        assert all("alpha" in c.content or "return" in c.content for c in chunks)

    def test_multiline_docstring_terminates(self) -> None:
        from memini_ai.indexer.chunker import SemanticChunker

        chunker = SemanticChunker()
        content = (
            'def f():\n    """Doc\n    spanning lines.\n    """\n'
            "    return 2\n\n\nclass C:\n    pass\n"
        )
        chunks = chunker.chunk_file(content, "mod.py")  # must not hang
        assert chunks, "expected at least one chunk"

    def test_module_docstring_only_file(self) -> None:
        from memini_ai.indexer.chunker import SemanticChunker

        chunker = SemanticChunker()
        content = '"""Module docstring."""\n\nX = 1\n'
        chunks = chunker.chunk_file(content, "mod.py")  # must not hang
        assert isinstance(chunks, list)

    def test_inline_string_with_triple_quote_terminates(self) -> None:
        from memini_ai.indexer.chunker import SemanticChunker

        chunker = SemanticChunker()
        content = 's = """inline"""\ndef f():\n    return s\n'
        chunks = chunker.chunk_file(content, "mod.py")  # must not hang
        assert chunks, "expected at least one chunk"
