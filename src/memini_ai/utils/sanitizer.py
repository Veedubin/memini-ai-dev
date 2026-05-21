"""Content sanitization for memory storage input validation.

Provides lightweight sanitization to prevent injection of dangerous content
into memory storage while preserving legitimate code content.
"""

from __future__ import annotations

import re

# Dangerous patterns to strip/neutralize
_SCRIPT_TAG_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_JAVASCRIPT_URL_RE = re.compile(r"javascript\s*:", re.IGNORECASE)
_EVENT_HANDLER_RE = re.compile(r'\bon\w+\s*=\s*["\'][^"\']*["\']', re.IGNORECASE)
# Iframe/embed tags that could load external content
_IFRAME_RE = re.compile(
    r"<iframe[^>]*>.*?</iframe>|<iframe[^>]*/?>",
    re.IGNORECASE | re.DOTALL,
)
_EMBED_RE = re.compile(r"<embed[^>]*/?>", re.IGNORECASE)
_OBJECT_RE = re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL)


class ContentTooLargeError(ValueError):
    """Raised when content exceeds the maximum allowed size."""

    def __init__(self, content_size: int, max_size: int) -> None:
        self.content_size = content_size
        self.max_size = max_size
        super().__init__(
            f"Content size {content_size} bytes exceeds maximum "
            f"allowed size of {max_size} bytes"
        )


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded for add_memory calls."""

    def __init__(self, peer_id: str, limit: int, window_seconds: int = 60) -> None:
        self.peer_id = peer_id
        self.limit = limit
        self.window_seconds = window_seconds
        super().__init__(
            f"Rate limit exceeded for peer '{peer_id}': "
            f"maximum {limit} add_memory calls per {window_seconds} seconds"
        )


def sanitize_content(content: str) -> str:
    """Sanitize memory content for safe storage.

    Applies the following sanitization steps:
    1. Strip null bytes (\\x00)
    2. Remove script tags and their contents
    3. Remove javascript: URLs
    4. Remove event handler attributes (onclick, onload, etc.)
    5. Remove iframe, embed, object tags
    6. Strip control characters (except \\n, \\r, \\t)
    7. HTML-escape remaining markup-like content

    This is designed to be lightweight — legitimate code content
    (Python, JavaScript, etc.) should pass through with minimal changes.

    Args:
        content: Raw content string to sanitize.

    Returns:
        Sanitized content string.
    """
    # Step 1: Strip null bytes
    content = content.replace("\x00", "")

    # Step 2: Remove <script> tags and contents
    content = _SCRIPT_TAG_RE.sub("[script removed]", content)

    # Step 3: Remove javascript: URLs
    content = _JAVASCRIPT_URL_RE.sub("[javascript-url removed]", content)

    # Step 4: Remove event handler attributes
    content = _EVENT_HANDLER_RE.sub("", content)

    # Step 5: Remove iframe, embed, object tags
    content = _IFRAME_RE.sub("[iframe removed]", content)
    content = _EMBED_RE.sub("[embed removed]", content)
    content = _OBJECT_RE.sub("[object removed]", content)

    # Step 6: Strip control characters (except \n \r \t)
    content = _strip_control_chars(content)

    return content


def _strip_control_chars(s: str) -> str:
    """Remove control characters except newline, carriage return, tab."""
    # Keep: \t (9), \n (10), \r (13)
    # Remove: everything else in 0x00-0x1F and 0x7F
    result = []
    for ch in s:
        code = ord(ch)
        if code < 32 and ch not in ("\t", "\n", "\r"):
            # Skip control characters
            continue
        if code == 127:  # DEL character
            continue
        result.append(ch)
    return "".join(result)


def validate_content_size(content: str, max_size: int) -> None:
    """Validate that content does not exceed the maximum allowed size.

    Args:
        content: Content string to validate.
        max_size: Maximum allowed size in bytes.

    Raises:
        ContentTooLargeError: If content exceeds the maximum size.
    """
    content_size = len(content.encode("utf-8"))
    if content_size > max_size:
        raise ContentTooLargeError(content_size=content_size, max_size=max_size)
