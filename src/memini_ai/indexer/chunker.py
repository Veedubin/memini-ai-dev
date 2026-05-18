"""Semantic chunker with language-aware parsing and sliding window."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    content: str
    path: str
    chunk_index: int
    total_chunks: int
    start_line: int
    end_line: int
    language: str | None = None

    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation: 4 chars per token)."""
        return len(self.content) // 4


@dataclass
class SemanticChunker:
    """Semantic + sliding window chunker for file content.

    Supports:
    - Semantic chunking at function/class boundaries
    - Sliding window with configurable size and overlap
    - Language-aware parsing for Python, JS, TS, etc.
    """

    _chunk_size: int = 512
    _chunk_overlap: int = 50
    _language_parsers: dict[str, Callable[[str], list[Chunk]]] = field(
        default_factory=dict, init=False
    )

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._register_parsers()

    def _register_parsers(self) -> None:
        """Register language-specific parsers."""
        self._language_parsers = {
            "py": self._parse_python,
            "python": self._parse_python,
            "js": self._parse_javascript,
            "javascript": self._parse_javascript,
            "ts": self._parse_typescript,
            "typescript": self._parse_typescript,
            "jsx": self._parse_typescript,
            "tsx": self._parse_typescript,
        }

    def _get_language(self, file_path: str) -> str | None:
        """Get language from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Language identifier or None.
        """
        ext = Path(file_path).suffix.lstrip(".")
        return ext if ext else None

    def chunk_file(self, content: str, file_path: str) -> list[Chunk]:
        """Chunk a file's content using semantic + sliding window approach.

        Args:
            content: File content as string.
            file_path: Path to the file (for language detection).

        Returns:
            List of Chunk objects.
        """
        language = self._get_language(file_path)
        parser = self._language_parsers.get(language) if language else None

        if parser:
            # Try semantic parsing first
            semantic_chunks = parser(content)
            if semantic_chunks:
                # Merge small chunks and apply sliding window
                return self._merge_and_window(semantic_chunks, file_path, language)

        # Fall back to sliding window
        return self._sliding_window(content, file_path, language)

    def _merge_and_window(
        self, semantic_chunks: list[Chunk], file_path: str, language: str | None
    ) -> list[Chunk]:
        """Merge small semantic chunks and apply sliding window.

        Args:
            semantic_chunks: Chunks from semantic parsing.
            file_path: File path for context.
            language: Language identifier.

        Returns:
            List of merged chunks with sliding window applied.
        """
        # Merge chunks that are too small
        merged: list[Chunk] = []
        current = ""
        current_start = 0
        current_lines = 0

        for chunk in semantic_chunks:
            # Simple merge strategy: if current + chunk fits in target, merge
            # Target is chunk_size * 4 chars (4 chars per token approximation)
            target_chars = self._chunk_size * 4

            if len(current) + len(chunk.content) <= target_chars:
                if current:
                    current += "\n"
                current += chunk.content
                if current_lines == 0:
                    current_start = chunk.start_line
                current_lines = chunk.end_line - current_start + 1
            else:
                if current:
                    merged.append(
                        Chunk(
                            content=current,
                            path=file_path,
                            chunk_index=len(merged),
                            total_chunks=0,  # Will be set later
                            start_line=current_start,
                            end_line=current_lines,
                            language=language,
                        )
                    )
                current = chunk.content
                current_start = chunk.start_line
                current_lines = chunk.end_line - chunk.start_line + 1

        if current:
            merged.append(
                Chunk(
                    content=current,
                    path=file_path,
                    chunk_index=len(merged),
                    total_chunks=0,
                    start_line=current_start,
                    end_line=current_lines,
                    language=language,
                )
            )

        # Apply sliding window
        return self._apply_sliding_window(merged, file_path, language)

    def _sliding_window(
        self, content: str, file_path: str, language: str | None
    ) -> list[Chunk]:
        """Apply sliding window chunking to content.

        Args:
            content: File content.
            file_path: File path for context.
            language: Language identifier.

        Returns:
            List of chunks with sliding window applied.
        """
        lines = content.split("\n")
        target_chars = self._chunk_size * 4
        overlap_chars = self._chunk_overlap * 4

        chunks: list[Chunk] = []
        start = 0

        while start < len(lines):
            end = start
            char_count = 0
            chunk_lines: list[str] = []

            # Grow chunk until we hit target size
            while end < len(lines):
                line = lines[end]
                line_len = len(line) + 1  # +1 for newline

                if char_count + line_len > target_chars and chunk_lines:
                    break

                chunk_lines.append(line)
                char_count += line_len
                end += 1

            if not chunk_lines:
                break

            chunk = Chunk(
                content="\n".join(chunk_lines),
                path=file_path,
                chunk_index=len(chunks),
                total_chunks=0,
                start_line=start + 1,
                end_line=end,
                language=language,
            )
            chunks.append(chunk)

            # Slide window with overlap
            # Move back by overlap_lines
            overlap_lines = overlap_chars // 60  # Approximate chars per line
            start = max(start + 1, end - overlap_lines)

        # Set total_chunks
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = total
            chunk.chunk_index = i

        return chunks

    def _apply_sliding_window(
        self, chunks: list[Chunk], file_path: str, language: str | None
    ) -> list[Chunk]:
        """Apply sliding window overlap to already-chunked content.

        Args:
            chunks: List of chunks to apply window to.
            file_path: File path for context.
            language: Language identifier.

        Returns:
            List of chunks with overlap applied.
        """
        if not chunks:
            return []

        overlap_chars = self._chunk_overlap * 4
        result = []
        prev_content = ""

        for _i, chunk in enumerate(chunks):
            # Prepend overlap from previous chunk
            if prev_content and overlap_chars > 0:
                # Take last overlap_chars from previous content
                overlap = prev_content[-overlap_chars:]
                if overlap:
                    chunk.content = overlap + "\n" + chunk.content

            prev_content = chunk.content
            result.append(chunk)

        # Set total_chunks
        total = len(result)
        for i, chunk in enumerate(result):
            chunk.total_chunks = total
            chunk.chunk_index = i

        return result

    def _parse_python(self, content: str) -> list[Chunk]:
        """Parse Python code into semantic chunks.

        Args:
            content: Python source code.

        Returns:
            List of chunks at function/class boundaries.
        """
        chunks = []
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_start = 1
        in_docstring = False
        docstring_delim = None

        # Patterns for Python structure
        class_pattern = re.compile(r"^class\s+\w+")
        func_pattern = re.compile(r"^(async\s+)?def\s+\w+")
        # Note: import_pattern and decorator_pattern reserved for future use
        _ = re.compile(r"^(import|from)\s+")  # noqa: F841
        _ = re.compile(r"^@\w+")  # noqa: F841

        i = 0
        while i < len(lines):
            line = lines[i]

            # Handle docstrings
            if not in_docstring:
                if '"""' in line or "'''" in line:
                    delim = '"""' if '"""' in line else "'''"
                    if delim in line:
                        # Single line docstring
                        if class_pattern.match(line) or func_pattern.match(line):
                            current_chunk.append(line)
                        continue
                    else:
                        in_docstring = True
                        docstring_delim = delim
                        if current_chunk:
                            chunks.append(
                                self._make_chunk(current_chunk, current_start)
                            )
                            current_chunk = []
                        current_start = i + 1
                elif class_pattern.match(line) or func_pattern.match(line):
                    if current_chunk:
                        chunks.append(self._make_chunk(current_chunk, current_start))
                        current_chunk = []
                    current_start = i + 1
            else:
                if docstring_delim and docstring_delim in line:
                    in_docstring = False
                    docstring_delim = None

            current_chunk.append(line)
            i += 1

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, current_start))

        return [c for c in chunks if c.content.strip()]

    def _parse_javascript(self, content: str) -> list[Chunk]:
        """Parse JavaScript/JSX into semantic chunks.

        Args:
            content: JavaScript source code.

        Returns:
            List of chunks at function/class boundaries.
        """
        chunks = []
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_start = 1
        in_block_comment = False

        # Patterns for JS structure
        class_pattern = re.compile(r"^class\s+\w+")
        func_pattern = re.compile(r"^(async\s+)?(function\s+\w+|const\s+\w+\s*=)")
        arrow_func_pattern = re.compile(r"^const\s+\w+\s*=\s*\(")
        # Note: import_pattern and decorator_pattern reserved for future use
        _ = re.compile(r"^(import|export)\s+")  # noqa: F841
        _ = re.compile(r"^@\w+")  # noqa: F841

        for i, line in enumerate(lines):
            # Handle block comments
            if not in_block_comment:
                if line.strip().startswith("/*"):
                    in_block_comment = True
                    if current_chunk:
                        chunks.append(self._make_chunk(current_chunk, current_start))
                        current_chunk = []
                    current_start = i + 1
                    continue
                elif (
                    class_pattern.match(line)
                    or func_pattern.match(line)
                    or arrow_func_pattern.match(line)
                ):
                    if current_chunk:
                        chunks.append(self._make_chunk(current_chunk, current_start))
                        current_chunk = []
                    current_start = i + 1
            else:
                if "*/" in line:
                    in_block_comment = False

            current_chunk.append(line)

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, current_start))

        return [c for c in chunks if c.content.strip()]

    def _parse_typescript(self, content: str) -> list[Chunk]:
        """Parse TypeScript/TSX into semantic chunks.

        Args:
            content: TypeScript source code.

        Returns:
            List of chunks at function/class boundaries.
        """
        # TypeScript uses same structure as JavaScript
        # but also has interface, type, enum
        chunks = []
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_start = 1
        in_block_comment = False

        # Patterns for TS structure
        class_pattern = re.compile(r"^class\s+\w+")
        func_pattern = re.compile(r"^(async\s+)?(function\s+\w+|const\s+\w+\s*=)")
        arrow_func_pattern = re.compile(r"^const\s+\w+\s*=\s*\(")
        interface_pattern = re.compile(r"^(interface|type|enum)\s+\w+")
        # Note: import_pattern reserved for future use
        _ = re.compile(r"^(import|export)\s+")  # noqa: F841

        for i, line in enumerate(lines):
            if not in_block_comment:
                if line.strip().startswith("/*"):
                    in_block_comment = True
                    if current_chunk:
                        chunks.append(self._make_chunk(current_chunk, current_start))
                        current_chunk = []
                    current_start = i + 1
                    continue
                elif (
                    class_pattern.match(line)
                    or func_pattern.match(line)
                    or arrow_func_pattern.match(line)
                    or interface_pattern.match(line)
                ):
                    if current_chunk:
                        chunks.append(self._make_chunk(current_chunk, current_start))
                        current_chunk = []
                    current_start = i + 1
            else:
                if "*/" in line:
                    in_block_comment = False

            current_chunk.append(line)

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, current_start))

        return [c for c in chunks if c.content.strip()]

    def _make_chunk(self, lines: list[str], start_line: int) -> Chunk:
        """Create a chunk from lines.

        Args:
            lines: Lines of content.
            start_line: Starting line number (1-indexed).

        Returns:
            Chunk object.
        """
        return Chunk(
            content="\n".join(lines),
            path="",  # Will be set later
            chunk_index=0,
            total_chunks=0,
            start_line=start_line,
            end_line=start_line + len(lines) - 1,
            language=None,
        )
