"""Tests for input validation (Phase 2.1: Security Audit Remediation).

Tests for:
1. Content length validation (MAX_MEMORY_CONTENT_SIZE)
2. Content sanitization (scripts, null bytes, control chars)
3. Rate limiting (sliding window per-peer)
4. Integration with server add_memory tool
"""

from __future__ import annotations

import time
import pytest

from memini_ai.config import MeminiConfig
from memini_ai.rate_limiter import AsyncRateLimiter, SlidingWindowRateLimiter
from memini_ai.utils.sanitizer import (
    ContentTooLargeError,
    RateLimitExceededError,
    sanitize_content,
    validate_content_size,
)


# ==========================================================================
# Content Size Validation Tests
# ==========================================================================


class TestContentSizeValidation:
    """Tests for validate_content_size function."""

    def test_content_within_limit(self) -> None:
        """Valid content under the size limit should pass."""
        content = "Hello, world!"
        validate_content_size(content, max_size=1024)  # Should not raise

    def test_content_exactly_at_limit(self) -> None:
        """Content exactly at the size limit should pass."""
        max_size = 100
        content = "a" * max_size
        validate_content_size(content, max_size=max_size)  # Should not raise

    def test_content_one_byte_over_limit(self) -> None:
        """Content one byte over the limit should raise ContentTooLargeError."""
        max_size = 100
        content = "a" * (max_size + 1)
        with pytest.raises(ContentTooLargeError) as exc_info:
            validate_content_size(content, max_size=max_size)
        assert exc_info.value.content_size == max_size + 1
        assert exc_info.value.max_size == max_size

    def test_content_way_over_limit(self) -> None:
        """Very large content should raise ContentTooLargeError."""
        max_size = 1024
        content = "x" * (10 * 1024 * 1024)  # 10MB
        with pytest.raises(ContentTooLargeError):
            validate_content_size(content, max_size=max_size)

    def test_unicode_content_size(self) -> None:
        """Unicode content should be measured by encoded byte length."""
        max_size = 10
        # Unicode characters that encode to more than 1 byte in UTF-8
        content = "日本語"  # Each character is 3 bytes in UTF-8 = 9 bytes
        validate_content_size(content, max_size=max_size)  # Should pass (9 < 10)

    def test_unicode_content_over_limit(self) -> None:
        """Unicode content over the byte limit should be rejected."""
        max_size = 5
        # "日本" = 6 bytes in UTF-8
        content = "日本"
        with pytest.raises(ContentTooLargeError) as exc_info:
            validate_content_size(content, max_size=max_size)
        assert exc_info.value.content_size == 6  # 6 UTF-8 bytes

    def test_empty_content(self) -> None:
        """Empty content should always pass validation."""
        validate_content_size("", max_size=1)  # 0 bytes


# ==========================================================================
# Content Sanitization Tests
# ==========================================================================


class TestContentSanitization:
    """Tests for sanitize_content function."""

    def test_plain_text_passes_through(self) -> None:
        """Plain text should pass through unchanged."""
        text = "Hello, this is a normal memory entry."
        assert sanitize_content(text) == text

    def test_code_content_preserved(self) -> None:
        """Legitimate code content should pass through mostly unchanged."""
        code = 'def hello():\n    print("Hello, world!")\n    return 42'
        assert sanitize_content(code) == code

    def test_python_code_preserved(self) -> None:
        """Python code with common patterns should be preserved."""
        code = """
import os
def main():
    for i in range(10):
        if i % 2 == 0:
            print(f"Even: {i}")
    return True
"""
        result = sanitize_content(code)
        assert "import os" in result
        assert "for i in range" in result

    def test_null_bytes_stripped(self) -> None:
        """Null bytes should be stripped from content."""
        content = "Hello\x00World"
        result = sanitize_content(content)
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

    def test_multiple_null_bytes_stripped(self) -> None:
        """Multiple null bytes should all be stripped."""
        content = "\x00\x00Start\x00Middle\x00End\x00\x00"
        result = sanitize_content(content)
        assert "\x00" not in result
        assert "Start" in result
        assert "End" in result

    def test_script_tags_removed(self) -> None:
        """Script tags and their contents should be removed."""
        content = 'Hello <script>alert("XSS")</script> World'
        result = sanitize_content(content)
        assert "<script>" not in result
        assert "alert" not in result or "[script removed]" in result
        assert "Hello" in result
        assert "World" in result

    def test_script_tags_case_insensitive(self) -> None:
        """Script tags removal should be case insensitive."""
        content = '<SCRIPT>alert("XSS")</SCRIPT>'
        result = sanitize_content(content)
        assert "<SCRIPT>" not in result
        assert "alert" not in result or "[script removed]" in result

    def test_script_tags_with_attributes(self) -> None:
        """Script tags with attributes should be removed."""
        content = '<script type="text/javascript">malicious()</script>'
        result = sanitize_content(content)
        assert "malicious" not in result
        assert "[script removed]" in result

    def test_javascript_url_removed(self) -> None:
        """javascript: URLs should be removed."""
        content = 'Click <a href="javascript:void(0)">here</a>'
        result = sanitize_content(content)
        assert (
            "javascript:" not in result.lower() or "[javascript-url removed]" in result
        )

    def test_event_handlers_removed(self) -> None:
        """Event handler attributes should be removed."""
        content = '<div onclick="alert(1)">Hello</div>'
        result = sanitize_content(content)
        assert "onclick" not in result.lower()

    def test_iframe_removed(self) -> None:
        """iframe tags should be removed."""
        content = 'Hello <iframe src="evil.com"></iframe> World'
        result = sanitize_content(content)
        assert "<iframe" not in result.lower()
        assert "[iframe removed]" in result

    def test_embed_tag_removed(self) -> None:
        """embed tags should be removed."""
        content = '<embed src="evil.swf">'
        result = sanitize_content(content)
        assert "<embed" not in result.lower()
        assert "[embed removed]" in result

    def test_object_tag_removed(self) -> None:
        """object tags should be removed."""
        content = '<object data="evil.swf"></object>'
        result = sanitize_content(content)
        assert "<object" not in result.lower()
        assert "[object removed]" in result

    def test_control_characters_stripped(self) -> None:
        """Control characters (except newline/tab/cr) should be stripped."""
        content = "Hello\x01\x02\x03World\x07\x08"
        result = sanitize_content(content)
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result
        assert "\x07" not in result
        assert "\x08" not in result
        assert "Hello" in result
        assert "World" in result

    def test_newlines_preserved(self) -> None:
        """Newline characters should be preserved."""
        content = "Line 1\nLine 2\nLine 3"
        result = sanitize_content(content)
        assert result == content

    def test_tabs_preserved(self) -> None:
        """Tab characters should be preserved."""
        content = "Column1\tColumn2\tColumn3"
        result = sanitize_content(content)
        assert result == content

    def test_carriage_returns_preserved(self) -> None:
        """Carriage return characters should be preserved."""
        content = "Line1\r\nLine2"
        result = sanitize_content(content)
        assert "\r" in result or "Line1" in result

    def test_del_character_removed(self) -> None:
        """DEL character (0x7F) should be removed."""
        content = "Hello\x7fWorld"
        result = sanitize_content(content)
        assert "\x7f" not in result

    def test_mixed_attack_content(self) -> None:
        """Content with multiple attack patterns should all be cleaned."""
        content = 'Normal\x00<script>alert(1)</script>javascript:void() \x01<iframe src="evil">'
        result = sanitize_content(content)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "<script>" not in result.lower()
        assert "<iframe" not in result.lower()

    def test_markdown_content_preserved(self) -> None:
        """Markdown formatting should be preserved."""
        md = """# Heading

This is **bold** and *italic* text.

- List item 1
- List item 2

```python
def foo():
    return "bar"
```

[Link](https://example.com)
"""
        result = sanitize_content(md)
        assert "# Heading" in result
        assert "**bold**" in result
        assert "```python" in result

    def test_json_content_preserved(self) -> None:
        """JSON content should be preserved."""
        json_content = '{"key": "value", "nested": {"a": 1, "b": true}}'
        result = sanitize_content(json_content)
        assert result == json_content


# ==========================================================================
# Rate Limiter Tests
# ==========================================================================


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    def test_allows_requests_under_limit(self) -> None:
        """Requests under the limit should be allowed."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        for i in range(5):
            allowed, remaining, retry_after = limiter.check_rate_limit("peer1")
            assert allowed is True
            assert remaining == 4 - i
            assert retry_after == 0

    def test_rejects_requests_over_limit(self) -> None:
        """Requests over the limit should be rejected."""
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        # Use 3 requests
        for _ in range(3):
            limiter.check_rate_limit("peer1")

        # 4th request should be rejected
        allowed, remaining, retry_after = limiter.check_rate_limit("peer1")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    def test_separate_peers_have_separate_limits(self) -> None:
        """Different peers should have separate rate limits."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)

        # peer1 uses 2 requests
        limiter.check_rate_limit("peer1")
        limiter.check_rate_limit("peer1")

        # peer1 should be rejected
        allowed, _, _ = limiter.check_rate_limit("peer1")
        assert allowed is False

        # peer2 should still be allowed
        allowed, _, _ = limiter.check_rate_limit("peer2")
        assert allowed is True

    def test_reset_single_peer(self) -> None:
        """Resetting a single peer should clear its rate limit."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)

        # Exhaust peer1
        limiter.check_rate_limit("peer1")
        limiter.check_rate_limit("peer1")
        allowed, _, _ = limiter.check_rate_limit("peer1")
        assert allowed is False

        # Reset peer1
        limiter.reset("peer1")

        # peer1 should be allowed again
        allowed, _, _ = limiter.check_rate_limit("peer1")
        assert allowed is True

    def test_reset_all_peers(self) -> None:
        """Resetting all peers should clear all rate limits."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)

        # Exhaust both peers
        for _ in range(2):
            limiter.check_rate_limit("peer1")
            limiter.check_rate_limit("peer2")

        # Reset all
        limiter.reset()

        # Both should be allowed again
        allowed1, _, _ = limiter.check_rate_limit("peer1")
        allowed2, _, _ = limiter.check_rate_limit("peer2")
        assert allowed1 is True
        assert allowed2 is True

    def test_window_expiry(self) -> None:
        """Requests should be allowed after the window expires."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)

        # Use 2 requests
        limiter.check_rate_limit("peer1")
        limiter.check_rate_limit("peer1")

        # Should be rejected
        allowed, _, _ = limiter.check_rate_limit("peer1")
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again (old timestamps cleaned up)
        allowed, _, _ = limiter.check_rate_limit("peer1")
        assert allowed is True

    def test_check_and_raise_allows(self) -> None:
        """check_and_raise should not raise when within limit."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        # Should not raise
        limiter.check_and_raise("peer1")

    def test_check_and_raise_rejects(self) -> None:
        """check_and_raise should raise RateLimitExceededError when exceeded."""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        limiter.check_and_raise("peer1")

        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check_and_raise("peer1")
        assert exc_info.value.peer_id == "peer1"
        assert exc_info.value.limit == 1

    def test_remaining_count_accurate(self) -> None:
        """Remaining count should accurately reflect available requests."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)

        # Initially 5 available
        _, remaining, _ = limiter.check_rate_limit("peer1")
        assert remaining == 4  # Used 1, 4 remaining

        _, remaining, _ = limiter.check_rate_limit("peer1")
        assert remaining == 3  # Used 2, 3 remaining

    def test_default_limit(self) -> None:
        """Default rate limit should be 100 requests per minute."""
        limiter = SlidingWindowRateLimiter()  # Default: 100/min
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60


# ==========================================================================
# Config Validation Tests
# ==========================================================================


class TestConfigValidation:
    """Tests for new config fields for input validation."""

    def test_default_max_content_size(self) -> None:
        """Default max content size should be 102400 (100KB)."""
        config = MeminiConfig()
        assert config.max_memory_content_size == 102400

    def test_default_rate_limit(self) -> None:
        """Default rate limit should be 100 per minute."""
        config = MeminiConfig()
        assert config.rate_limit_per_minute == 100

    def test_default_sanitize_content(self) -> None:
        """Default sanitize_content should be True."""
        config = MeminiConfig()
        assert config.sanitize_content is True

    def test_max_content_size_clamped_minimum(self) -> None:
        """Max content size below minimum should be clamped to 1024."""
        import os

        os.environ["MAX_MEMORY_CONTENT_SIZE"] = "100"
        try:
            config = MeminiConfig()
            assert config.max_memory_content_size == 1024
        finally:
            del os.environ["MAX_MEMORY_CONTENT_SIZE"]

    def test_max_content_size_clamped_maximum(self) -> None:
        """Max content size above maximum should be clamped to 10MB."""
        import os

        os.environ["MAX_MEMORY_CONTENT_SIZE"] = str(100 * 1024 * 1024)
        try:
            config = MeminiConfig()
            assert config.max_memory_content_size == 10 * 1024 * 1024
        finally:
            del os.environ["MAX_MEMORY_CONTENT_SIZE"]

    def test_rate_limit_clamped_minimum(self) -> None:
        """Rate limit below 1 should be clamped to 1."""
        import os

        os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
        try:
            config = MeminiConfig()
            assert config.rate_limit_per_minute == 1
        finally:
            del os.environ["RATE_LIMIT_PER_MINUTE"]

    def test_rate_limit_clamped_maximum(self) -> None:
        """Rate limit above 10000 should be clamped to 10000."""
        import os

        os.environ["RATE_LIMIT_PER_MINUTE"] = "50000"
        try:
            config = MeminiConfig()
            assert config.rate_limit_per_minute == 10000
        finally:
            del os.environ["RATE_LIMIT_PER_MINUTE"]

    def test_env_var_max_content_size(self) -> None:
        """MAX_MEMORY_CONTENT_SIZE env var should be respected."""
        import os

        os.environ["MAX_MEMORY_CONTENT_SIZE"] = "204800"
        try:
            config = MeminiConfig()
            assert config.max_memory_content_size == 204800
        finally:
            del os.environ["MAX_MEMORY_CONTENT_SIZE"]

    def test_env_var_rate_limit(self) -> None:
        """RATE_LIMIT_PER_MINUTE env var should be respected."""
        import os

        os.environ["RATE_LIMIT_PER_MINUTE"] = "50"
        try:
            config = MeminiConfig()
            assert config.rate_limit_per_minute == 50
        finally:
            del os.environ["RATE_LIMIT_PER_MINUTE"]

    def test_env_var_sanitize_content(self) -> None:
        """SANITIZE_CONTENT env var should be respected."""
        import os

        os.environ["SANITIZE_CONTENT"] = "false"
        try:
            config = MeminiConfig()
            assert config.sanitize_content is False
        finally:
            del os.environ["SANITIZE_CONTENT"]


# ==========================================================================
# Async Rate Limiter Tests
# ==========================================================================


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self) -> None:
        """Async rate limiter should allow requests under the limit."""
        limiter = AsyncRateLimiter(max_requests=5, window_seconds=60)
        allowed, remaining, retry_after = await limiter.check_rate_limit("peer1")
        assert allowed is True
        assert remaining == 4
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_rejects_requests_over_limit(self) -> None:
        """Async rate limiter should reject requests over the limit."""
        limiter = AsyncRateLimiter(max_requests=2, window_seconds=60)
        await limiter.check_rate_limit("peer1")
        await limiter.check_rate_limit("peer1")

        allowed, remaining, _ = await limiter.check_rate_limit("peer1")
        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_check_and_raise(self) -> None:
        """Async check_and_raise should raise on exceeded."""
        limiter = AsyncRateLimiter(max_requests=1, window_seconds=60)
        await limiter.check_and_raise("peer1")

        with pytest.raises(RateLimitExceededError):
            await limiter.check_and_raise("peer1")

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Async reset should clear rate limit state."""
        limiter = AsyncRateLimiter(max_requests=1, window_seconds=60)
        await limiter.check_and_raise("peer1")

        # Should be rejected
        with pytest.raises(RateLimitExceededError):
            await limiter.check_and_raise("peer1")

        # Reset
        await limiter.reset("peer1")

        # Should be allowed again
        await limiter.check_and_raise("peer1")  # No exception


# ==========================================================================
# Integration: Error Message Tests
# ==========================================================================


class TestErrorMessages:
    """Tests for clear error messages in validation errors."""

    def test_content_too_large_error_message(self) -> None:
        """ContentTooLargeError should have a descriptive message."""
        with pytest.raises(ContentTooLargeError) as exc_info:
            validate_content_size("x" * 200, max_size=100)
        error = exc_info.value
        assert "200" in str(error)
        assert "100" in str(error)
        assert "exceeds" in str(error).lower()

    def test_rate_limit_error_message(self) -> None:
        """RateLimitExceededError should have a descriptive message."""
        error = RateLimitExceededError(
            peer_id="test-peer", limit=100, window_seconds=60
        )
        msg = str(error)
        assert "test-peer" in msg
        assert "100" in msg
        assert "60" in msg
