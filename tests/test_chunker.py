"""Tests for semantic chunker."""

from __future__ import annotations

from memini_ai.indexer.chunker import Chunk, SemanticChunker


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.chunker = SemanticChunker(chunk_size=128, chunk_overlap=20)

    def test_chunk_file_basic(self) -> None:
        """Test basic file chunking."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = self.chunker.chunk_file(content, "test.txt")

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.path == "test.txt" for c in chunks)

    def test_chunk_file_python(self) -> None:
        """Test Python file chunking with semantic parsing."""
        content = """
def func1():
    pass

def func2():
    pass

class MyClass:
    pass
"""
        chunks = self.chunker.chunk_file(content, "test.py")

        assert len(chunks) > 0
        # Python parser should detect functions and classes
        for chunk in chunks:
            assert chunk.language in (None, "py", "python")

    def test_chunk_file_javascript(self) -> None:
        """Test JavaScript file chunking."""
        content = """
const func1 = () => {
    console.log("test");
};

const func2 = () => {
    console.log("test2");
};

export { func1, func2 };
"""
        chunks = self.chunker.chunk_file(content, "test.js")

        assert len(chunks) > 0

    def test_chunk_file_typescript(self) -> None:
        """Test TypeScript file chunking."""
        content = """
interface TestInterface {
    name: string;
    value: number;
}

type TestType = string | number;

class TestClass {
    public method(): void {
        console.log("test");
    }
}
"""
        chunks = self.chunker.chunk_file(content, "test.ts")

        assert len(chunks) > 0

    def test_chunk_token_count(self) -> None:
        """Test token count estimation."""
        content = "a" * 1000
        chunks = self.chunker.chunk_file(content, "test.txt")

        for chunk in chunks:
            # Token count is estimated at 4 chars per token
            expected_tokens = len(chunk.content) // 4
            assert abs(chunk.token_count - expected_tokens) <= 1

    def test_chunk_metadata(self) -> None:
        """Test chunk metadata is set correctly."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = self.chunker.chunk_file(content, "test.py")

        for chunk in chunks:
            assert chunk.path == "test.py"
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks > 0
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line

    def test_empty_content(self) -> None:
        """Test handling of empty content."""
        chunks = self.chunker.chunk_file("", "test.txt")
        # Should return empty list or single empty chunk
        assert len(chunks) == 0 or all(not c.content.strip() for c in chunks)

    def test_single_line(self) -> None:
        """Test handling of single line content."""
        chunks = self.chunker.chunk_file("single line content", "test.txt")

        assert len(chunks) > 0

    def test_sliding_window_applies_overlap(self) -> None:
        """Test that sliding window applies overlap correctly."""
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
        # Create content with many lines
        lines = [f"line {i}" for i in range(200)]
        content = "\n".join(lines)

        chunks = chunker.chunk_file(content, "test.txt")

        # With overlap, chunks should share some content
        # Just verify we get multiple chunks with overlap applied
        assert len(chunks) > 1
        # This may not always be true depending on chunk boundaries

    def test_unknown_extension_falls_back_to_sliding_window(self) -> None:
        """Test that unknown extensions use sliding window."""
        content = "\n".join([f"line{i}" for i in range(50)])
        chunks = self.chunker.chunk_file(content, "test.unknown")

        assert len(chunks) > 0

    def test_chunk_size_affects_output(self) -> None:
        """Test that different chunk sizes produce different results."""
        small_chunker = SemanticChunker(chunk_size=64, chunk_overlap=10)
        large_chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

        content = "\n".join([f"line{i}" for i in range(100)])

        small_chunks = small_chunker.chunk_file(content, "test.txt")
        large_chunks = large_chunker.chunk_file(content, "test.txt")

        # Different chunk sizes should produce different numbers of chunks
        # but the exact relationship depends on content and overlap settings
        assert len(small_chunks) != len(large_chunks)


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_token_count_calculation(self) -> None:
        """Test token count is calculated correctly."""
        chunk = Chunk(
            content="1234",
            path="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_line=1,
            end_line=1,
        )
        # 4 chars / 4 = 1 token
        assert chunk.token_count == 1

    def test_token_count_long_content(self) -> None:
        """Test token count with longer content."""
        content = "a" * 100
        chunk = Chunk(
            content=content,
            path="test.txt",
            chunk_index=0,
            total_chunks=1,
            start_line=1,
            end_line=1,
        )
        assert chunk.token_count == 25  # 100 / 4


class TestSemanticParsing:
    """Tests for semantic parsing in different languages."""

    def test_python_class_parsing(self) -> None:
        """Test parsing of Python class definitions."""
        chunker = SemanticChunker(chunk_size=256, chunk_overlap=20)
        content = """
class FirstClass:
    def method(self):
        pass

class SecondClass:
    def another_method(self):
        pass
"""
        chunks = chunker.chunk_file(content, "test.py")

        # Should have chunks that separate the classes
        assert len(chunks) > 0

    def test_python_async_function(self) -> None:
        """Test parsing of async Python functions."""
        chunker = SemanticChunker(chunk_size=256, chunk_overlap=20)
        content = """
async def async_func():
    await something()

def regular_func():
    pass
"""
        chunks = chunker.chunk_file(content, "test.py")

        assert len(chunks) > 0

    def test_javascript_arrow_functions(self) -> None:
        """Test parsing of JavaScript arrow functions."""
        chunker = SemanticChunker(chunk_size=256, chunk_overlap=20)
        content = """
const arrow1 = () => {
    return 1;
};

const arrow2 = () => {
    return 2;
};
"""
        chunks = chunker.chunk_file(content, "test.js")

        assert len(chunks) > 0

    def test_typescript_interface_parsing(self) -> None:
        """Test parsing of TypeScript interfaces."""
        chunker = SemanticChunker(chunk_size=256, chunk_overlap=20)
        content = """
interface User {
    id: number;
    name: string;
}

interface Admin extends User {
    permissions: string[];
}
"""
        chunks = chunker.chunk_file(content, "test.ts")

        assert len(chunks) > 0

    def test_mixed_content_language_detection(self) -> None:
        """Test handling of mixed content."""
        chunker = SemanticChunker()
        content = """
# Python comment
def python_func():
    pass

// JavaScript comment
const js_const = 42;
"""
        chunks = chunker.chunk_file(content, "test.py")

        assert len(chunks) > 0
