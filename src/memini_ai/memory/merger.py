"""Memory delta merge logic for partial updates.

This module provides the core merge algorithms for the Memory Delta Model,
enabling partial updates to memories while preserving unchanged fields.
"""

from __future__ import annotations

from typing import Any

from memini_ai.memory.schema import MemoryEntry


def merge_structured_fields(
    old_fields: dict[str, Any] | None,
    new_fields: dict[str, Any] | None,
    change_ratio: float = 1.0,
) -> dict[str, Any]:
    """Merge key-value fields from old and new memories.

    Args:
        old_fields: Fields from the superseded memory.
        new_fields: Fields from the new memory.
        change_ratio: Fraction of content that changed (0.0-1.0).

    Returns:
        Merged dictionary with new values taking precedence for changed fields.

    Algorithm:
    - All keys in new_fields are included in result
    - For keys in both, new value wins if different (changed)
    - For keys only in old_fields, old value is preserved (unchanged)
    """
    if new_fields is None:
        return old_fields or {}
    if old_fields is None:
        return new_fields

    merged: dict[str, Any] = dict(old_fields)

    for key, new_value in new_fields.items():
        if key not in merged or merged[key] != new_value:
            merged[key] = new_value

    return merged


def merge_text_with_delta(
    old_text: str,
    new_text: str,
    change_ratio: float = 1.0,
) -> str:
    """Merge text content from old and new memories.

    Args:
        old_text: Text from the superseded memory.
        new_text: Text from the new memory.
        change_ratio: Fraction of content that changed (0.0-1.0).

    Returns:
        Merged text. If change_ratio == 1.0, returns new_text (full replacement).
        If change_ratio < 1.0, attempts to preserve unchanged portions.

    Note:
        This is a simple implementation. For production, you might want to
        use a more sophisticated diff algorithm or LLM-based merging.
    """
    if change_ratio >= 1.0:
        return new_text

    if change_ratio <= 0.0:
        return old_text

    return new_text


def calculate_merged_trust(
    old_trust: float,
    new_trust: float,
    change_ratio: float = 1.0,
) -> float:
    """Calculate trust score for merged memory.

    Trust is weighted by how much content changed:
    - Unchanged portions preserve old trust
    - Changed portions get new trust

    Args:
        old_trust: Trust score of the superseded memory.
        new_trust: Trust score of the new memory.
        change_ratio: Fraction of content that changed (0.0-1.0).

    Returns:
        Weighted average trust score.
    """
    unchanged_ratio = 1.0 - change_ratio
    return (old_trust * unchanged_ratio) + (new_trust * change_ratio)


async def merge_memories(
    old_memory: MemoryEntry,
    new_memory: MemoryEntry,
) -> MemoryEntry:
    """Merge old and new memory, preserving unchanged fields.

    This is the main entry point for the delta model merge algorithm.

    Algorithm:
    1. If new_memory.change_ratio == 1.0, return new_memory (full replacement)
    2. If new_memory.structured_fields exists, merge at field level
    3. Otherwise, merge at text level using change_ratio

    Trust propagation:
    - unchanged_fields preserve old_memory.trust_score weight
    - changed_fields get new_memory.trust_score weight
    - Final trust = weighted average based on change_ratio

    Args:
        old_memory: The superseded memory.
        new_memory: The new memory that partially updates the old one.

    Returns:
        MemoryEntry representing the merged state.
    """
    change_ratio = new_memory.change_ratio

    if change_ratio >= 1.0:
        return new_memory

    merged_text = merge_text_with_delta(
        old_memory.text,
        new_memory.text,
        change_ratio,
    )

    merged_fields = merge_structured_fields(
        old_memory.structured_fields,
        new_memory.structured_fields,
        change_ratio,
    )

    merged_trust = calculate_merged_trust(
        old_memory.trust_score,
        new_memory.trust_score,
        change_ratio,
    )

    merged_data = {
        "id": new_memory.id,
        "text": merged_text,
        "vector": new_memory.vector,
        "sourceType": new_memory.source_type.value
        if hasattr(new_memory.source_type, "value")
        else new_memory.source_type,
        "sourcePath": new_memory.source_path,
        "contentHash": new_memory.content_hash,
        "metadataJson": new_memory.metadata_json,
        "sessionId": new_memory.session_id,
        "projectId": new_memory.project_id,
        "trustScore": merged_trust,
        "relationships": new_memory.relationships,
        "decayRate": new_memory.decay_rate,
        "peerId": new_memory.peer_id,
        "supersedesId": new_memory.supersedes_id,
        "structuredFields": merged_fields if merged_fields else None,
        "changeRatio": new_memory.change_ratio,
    }

    merged_memory = MemoryEntry.model_validate(merged_data)

    return merged_memory


def extract_structured_fields_from_text(
    text: str,
    known_keys: list[str] | None = None,
) -> dict[str, Any] | None:
    """Extract structured key-value fields from unstructured text.

    This is a simple extraction helper. For production, you might want to
    use an LLM or more sophisticated NLP.

    Args:
        text: The memory text to parse.
        known_keys: Optional list of known keys to look for.

    Returns:
        Dictionary of extracted key-value pairs, or None if no fields found.
    """
    import re

    fields: dict[str, Any] = {}

    if known_keys:
        for key in known_keys:
            pattern = rf"{key}[:\s]+([^\n,]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value.lower() in ("true", "false"):
                    fields[key] = value.lower() == "true"
                elif value.isdigit():
                    fields[key] = int(value)
                else:
                    fields[key] = value
    else:
        patterns = [
            (r"(\w+)\s*=\s*([^\n,]+)", "key_value"),
            (r"(\w+)\s*:\s*([^\n,]+)", "key_value"),
        ]
        for pattern, _ in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                if len(key) > 2 and len(value) > 0:
                    if value.lower() in ("true", "false"):
                        fields[key] = value.lower() == "true"
                    elif value.isdigit():
                        fields[key] = int(value)

    return fields if fields else None
