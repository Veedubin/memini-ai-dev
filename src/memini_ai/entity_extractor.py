"""Entity Extractor - Regex-based named entity extraction for knowledge graph.

Provides lightweight NER-style extraction using regex patterns and heuristics.
Designed for code/document context where entities like function names,
variables, and project names are more relevant than person names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from memini_ai.knowledge_graph import EntityType


@dataclass
class ExtractedEntity:
    """An entity extracted from text with metadata."""

    name: str  # Surface form as found in text
    type: EntityType  # Entity type classification
    confidence: float  # Extraction confidence 0.0-1.0
    start_pos: int  # Start position in text
    end_pos: int  # End position in text
    pattern_matched: str  # Which pattern matched


class EntityExtractor:
    """Regex-based entity extractor with heuristic resolution.

    Extracts named entities from text using regex patterns optimized for:
    - Code entities (function names, class names, variables, file paths)
    - Project entities (repo names, project names)
    - Organization entities (company names, team names)
    - Person names (proper case names)
    - Concept entities (multi-word capitalized phrases)

    Features:
    - Multi-pass extraction with pattern优先级
    - Confidence scoring based on pattern specificity
    - Basic entity resolution (canonical form)
    - Deduplication across patterns
    """

    # Pattern definitions with confidence weights
    PATTERNS: list[tuple[str, EntityType, float, str]] = [
        # File paths: highest confidence (unambiguous)
        (
            r'\b([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)\b',
            EntityType.CODE,
            0.9,
            "file_extension",
        ),
        # Project/Repo names: GitHub-style
        (
            r'\b([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)\b',
            EntityType.PROJECT,
            0.85,
            "repo_path",
        ),
        # Function/method names (snake_case)
        (
            r'\b([a-z][a-z0-9_]{2,30})\s*\(',
            EntityType.CODE,
            0.8,
            "function_call",
        ),
        # Function definitions (snake_case)
        (
            r'def\s+([a-z][a-z0-9_]{2,30})',
            EntityType.CODE,
            0.85,
            "function_def",
        ),
        # Class names (PascalCase)
        (
            r'\b([A-Z][a-zA-Z0-9]{2,30})\b',
            EntityType.CODE,
            0.75,
            "class_name",
        ),
        # Constants (SCREAMING_SNAKE_CASE)
        (
            r'\b([A-Z][A-Z0-9_]{2,30})\b',
            EntityType.CODE,
            0.7,
            "constant_name",
        ),
        # Variables (camelCase)
        (
            r'\b([a-z][a-zA-Z0-9]{2,30})\b',
            EntityType.CODE,
            0.5,
            "camel_case",
        ),
        # URLs
        (
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            EntityType.PROJECT,
            0.9,
            "url",
        ),
        # Organization names (Company Inc/LLC/Ltd/Corp)
        (
            r'\b([A-Z][a-zA-Z0-9\s]{2,40})\s+(Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Co\.?)\b',
            EntityType.ORGANIZATION,
            0.85,
            "company_suffix",
        ),
        # Organization names (The Company pattern)
        (
            r'\b[Tt]he\s+([A-Z][a-zA-Z0-9\s]{2,30})\s+(Team|Group|Company|Organization)\b',
            EntityType.ORGANIZATION,
            0.8,
            "the_team_pattern",
        ),
        # Person names (First Last pattern)
        (
            r'\b([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\b',
            EntityType.PERSON,
            0.7,
            "person_name",
        ),
        # Quoted strings as potential entities
        (
            r'"([A-Z][a-zA-Z0-9\s]{2,50})"',
            EntityType.CONCEPT,
            0.6,
            "quoted_phrase",
        ),
        # Single quoted strings
        (
            r"'([^']{2,50})'",
            EntityType.CONCEPT,
            0.5,
            "single_quoted",
        ),
        # Multi-word concepts (Title Case)
        (
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b',
            EntityType.CONCEPT,
            0.55,
            "title_case_concept",
        ),
        # Database table names (snake_case)
        (
            r'\b([a-z][a-z0-9_]{2,30}_[a-z][a-z0-9_]{2,30})\b',
            EntityType.CODE,
            0.75,
            "table_name",
        ),
        # Environment variables
        (
            r'\$([A-Z][A-Z0-9_]{2,30})\b',
            EntityType.CODE,
            0.85,
            "env_var",
        ),
        # Command line flags
        (
            r'(?:-{1,2}[a-z][a-z0-9\-]{1,20})',
            EntityType.CODE,
            0.6,
            "cli_flag",
        ),
    ]

    # Minimum length for entity names
    MIN_LENGTH = 2
    MAX_LENGTH = 100

    # Stop words to filter out
    STOP_WORDS: set[str] = {
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "could", "may", "might", "must", "can", "this", "that",
        "these", "those", "a", "an", "if", "else", "when", "while", "then",
    }

    # Common false positives to filter
    FALSE_POSITIVES: set[str] = {
        "True", "False", "None", "NULL", "undefined",
        "TODO", "FIXME", "NOTE", "DEBUG",
        "default", "import", "export", "return", "class",
        "function", "var", "let", "const", "static",
    }

    def __init__(self) -> None:
        """Initialize entity extractor with compiled patterns."""
        self._compiled_patterns: list[tuple[re.Pattern[str], EntityType, float, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for pattern, entity_type, confidence, pattern_name in self.PATTERNS:
            compiled = re.compile(pattern, re.MULTILINE)
            self._compiled_patterns.append((compiled, entity_type, confidence, pattern_name))

    def extract_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Input text to extract entities from.

        Returns:
            List of dicts with name, type, confidence, and pattern_matched.
        """
        seen: dict[str, dict[str, Any]] = {}
        positions: dict[str, tuple[int, int]] = {}

        for compiled, entity_type, base_confidence, pattern_name in self._compiled_patterns:
            for match in compiled.finditer(text):
                name = match.group(1) if match.lastindex else match.group(0)

                if not name or len(name) < self.MIN_LENGTH or len(name) > self.MAX_LENGTH:
                    continue

                # Skip stop words and false positives
                if name.lower() in self.STOP_WORDS or name in self.FALSE_POSITIVES:
                    continue

                # Skip if mostly numbers
                alpha_ratio = sum(c.isalpha() for c in name) / len(name)
                if alpha_ratio < 0.3:
                    continue

                # Calculate confidence adjustment
                confidence = self._adjust_confidence(name, entity_type, base_confidence)

                key = name.lower()
                start_pos = match.start(1) if match.lastindex else match.start()
                end_pos = match.end(1) if match.lastindex else match.end()

                # Keep highest confidence match for deduplication
                if key not in seen or confidence > seen[key]["confidence"]:
                    seen[key] = {
                        "name": name,
                        "type": entity_type,
                        "confidence": confidence,
                        "pattern_matched": pattern_name,
                        "mentions": [name],
                    }
                    positions[key] = (start_pos, end_pos)

        # Convert to list and sort by confidence
        results = list(seen.values())
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return results

    def _adjust_confidence(
        self,
        name: str,
        entity_type: EntityType,
        base_confidence: float,
    ) -> float:
        """Adjust confidence based on heuristics.

        Args:
            name: Entity name.
            entity_type: Detected entity type.
            base_confidence: Base confidence from pattern.

        Returns:
            Adjusted confidence 0.0-1.0.
        """
        confidence = base_confidence

        # Longer names are more likely to be real entities
        if len(name) > 10:
            confidence += 0.05
        elif len(name) > 20:
            confidence += 0.1

        # Names with mixed case are more likely to be intentional
        has_upper = any(c.isupper() for c in name)
        has_lower = any(c.islower() for c in name)
        if has_upper and has_lower:
            confidence += 0.05

        # Penalize very short names
        if len(name) <= 3:
            confidence -= 0.2

        # Code entities get confidence boost if they look like real identifiers
        if entity_type == EntityType.CODE:
            if re.match(r'^[a-z][a-z0-9_]+$', name):  # snake_case
                confidence += 0.1
            elif re.match(r'^[A-Z][a-zA-Z0-9]+$', name):  # PascalCase
                confidence += 0.1
            elif re.match(r'^[A-Z][A-Z0-9_]+$', name):  # SCREAMING_SNAKE
                confidence += 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def extract_with_context(
        self,
        text: str,
        context_window: int = 50,
    ) -> list[dict[str, Any]]:
        """Extract entities with surrounding context.

        Args:
            text: Input text.
            context_window: Characters of context to include.

        Returns:
            List of entity dicts with context included.
        """
        entities = self.extract_from_text(text)

        for entity in entities:
            # Find position in text
            name = entity["name"]
            start = text.find(name)
            if start >= 0:
                end = start + len(name)
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                entity["context"] = text[ctx_start:ctx_end]
                entity["position"] = {"start": start, "end": end}

        return entities

    def resolve_canonical_form(self, name: str) -> str:
        """Resolve a surface form to canonical form.

        Applies heuristics to normalize entity names:
        - Lowercase for code entities
        - Title case for concepts/persons
        - Keep original case for organizations

        Args:
            name: Surface form as found in text.

        Returns:
            Canonical form of the entity.
        """
        if not name:
            return name

        # For code entities, canonical is lowercase
        # Check if it looks like code
        if re.match(r'^[a-z][a-z0-9_]+$', name):
            return name.lower()
        elif re.match(r'^[A-Z][a-zA-Z0-9]+$', name):
            return name  # PascalCase stays as-is
        elif re.match(r'^[A-Z][A-Z0-9_]+$', name):
            return name  # SCREAMING_SNAKE stays as-is

        # For natural language, title case
        words = name.split()
        if len(words) > 1:
            return " ".join(w.capitalize() for w in words)
        return name.capitalize()

    def filter_by_type(
        self,
        entities: list[dict[str, Any]],
        entity_type: EntityType,
    ) -> list[dict[str, Any]]:
        """Filter entities by type.

        Args:
            entities: List of entity dicts.
            entity_type: Type to filter by.

        Returns:
            Filtered list of entities.
        """
        return [e for e in entities if e["type"] == entity_type]

    def get_type_distribution(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Get distribution of entity types.

        Args:
            entities: List of entity dicts.

        Returns:
            Dictionary mapping type to count.
        """
        distribution: dict[str, int] = {}
        for entity in entities:
            type_key = entity["type"].value
            distribution[type_key] = distribution.get(type_key, 0) + 1
        return distribution

    def deduplicate_by_similarity(
        self,
        entities: list[dict[str, Any]],
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Deduplicate entities based on similarity.

        Uses simple string similarity (length + substring matching).

        Args:
            entities: List of entity dicts.
            threshold: Similarity threshold (0.0-1.0).

        Returns:
            Deduplicated list of entities.
        """
        if not entities:
            return []

        # Sort by confidence descending
        sorted_entities = sorted(entities, key=lambda x: x["confidence"], reverse=True)
        result: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for entity in sorted_entities:
            name_lower = entity["name"].lower()

            # Check against already selected entities
            is_duplicate = False
            for seen in seen_names:
                # Exact match
                if name_lower == seen:
                    is_duplicate = True
                    break
                # One contains the other
                if len(name_lower) > 3 and len(seen) > 3:
                    if name_lower in seen or seen in name_lower:
                        is_duplicate = True
                        break

            if not is_duplicate:
                result.append(entity)
                seen_names.add(name_lower)

        return result