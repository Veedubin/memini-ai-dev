"""Tests for hash utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from memini_ai.utils.hash import hash_content, hash_file


class TestHashContent:
    """Tests for hash_content function."""

    def test_hash_content_returns_hex_string(self) -> None:
        """Should return a hexadecimal string."""
        result = hash_content("test content")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex characters

    def test_hash_content_deterministic(self) -> None:
        """Same content should produce same hash."""
        content = "test content"
        result1 = hash_content(content)
        result2 = hash_content(content)
        assert result1 == result2

    def test_hash_content_different_inputs_different_hashes(self) -> None:
        """Different content should produce different hashes."""
        hash1 = hash_content("content a")
        hash2 = hash_content("content b")
        assert hash1 != hash2

    def test_hash_content_unicode(self) -> None:
        """Should handle unicode content correctly."""
        result = hash_content("Hello 世界 🌍")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_content_empty_string(self) -> None:
        """Should handle empty string."""
        result = hash_content("")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_content_long_content(self) -> None:
        """Should handle long content."""
        long_content = "x" * 1_000_000
        result = hash_content(long_content)
        assert isinstance(result, str)
        assert len(result) == 64


class TestHashFile:
    """Tests for hash_file function."""

    def test_hash_file_returns_hex_string(self, tmp_path: Path) -> None:
        """Should return a hexadecimal string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = hash_file(test_file)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_file_deterministic(self, tmp_path: Path) -> None:
        """Same file content should produce same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result1 = hash_file(test_file)
        result2 = hash_file(test_file)
        assert result1 == result2

    def test_hash_file_nonexistent_raises(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/path/file.txt")

    def test_hash_file_directory_raises(self, tmp_path: Path) -> None:
        """Should raise IsADirectoryError for directory path."""
        with pytest.raises(IsADirectoryError):
            hash_file(tmp_path)

    def test_hash_file_different_content_different_hashes(self, tmp_path: Path) -> None:
        """Different file content should produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content a")
        file2.write_text("content b")
        hash1 = hash_file(file1)
        hash2 = hash_file(file2)
        assert hash1 != hash2

    def test_hash_file_binary_content(self, tmp_path: Path) -> None:
        """Should handle binary content correctly."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04")
        result = hash_file(test_file)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_file_unicode_content(self, tmp_path: Path) -> None:
        """Should handle unicode content correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello 世界 🌍", encoding="utf-8")
        result = hash_file(test_file)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_file_string_path(self, tmp_path: Path) -> None:
        """Should accept string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        result = hash_file(str(test_file))
        assert isinstance(result, str)
        assert len(result) == 64
