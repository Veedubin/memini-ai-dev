"""Dialectic Engine - Argumentation engine for resolving memory contradictions.

Phase 4D of Memini-ai v3.0 - builds reasoned pro/con cases for conflicting memories
using LLM-based dialectic reasoning. Helps resolve contradictions in the memory
graph by generating arguments for each side.

Features:
- Contradiction detection via CONTRADICTS relationships
- Argument generation (pro/con for each side) using LLM
- Resolution synthesis from dialectic arguments
- Challenge/response workflow for ongoing dialectic
- All features optional (DIALECTIC_ENABLED=false disables)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx

from memini_ai.config import get_config
from memini_ai.memory.schema import RelationshipType
from memini_ai.utils.logger import logger

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem

# LLM prompts for dialectic reasoning
DIALECTIC_ARGUMENT_PROMPT = """
You are analyzing two contradictory memories and generating arguments for each side.

## Memory A (id: {memory_a_id}):
{memory_a_text}

## Memory B (id: {memory_b_id}):
{memory_b_text}

Your task is to analyze these memories and generate:

1. **Arguments supporting Memory A** (pro-A):
   - Why is Memory A likely to be correct?
   - What evidence or reasoning supports it?
   - Consider: source credibility, temporal factors, internal consistency

2. **Arguments supporting Memory B** (pro-B):
   - Why is Memory B likely to be correct?
   - What evidence or reasoning supports it?
   - Consider: source credibility, temporal factors, internal consistency

3. **Quality assessment** for each:
   - How confident are you in each memory's accuracy?
   - Are there potential confusions or misinterpretations?

Return JSON with the following structure:
{{
  "pro_a_arguments": [
    {{
      "argument": "description of the argument",
      "confidence": 0.0-1.0,
      "evidence": ["evidence1", "evidence2"]
    }}
  ],
  "pro_b_arguments": [
    {{
      "argument": "description of the argument",
      "confidence": 0.0-1.0,
      "evidence": ["evidence1", "evidence2"]
    }}
  ],
  "analysis": "overall analysis of the contradiction",
  "preferred_memory": "A" or "B" or "neither",
  "confidence": 0.0-1.0
}}

Return valid JSON only, no markdown or explanation.
"""

DIALECTIC_RESOLUTION_PROMPT = """
You have been presented with a dialectic analysis of contradictory memories.

## Arguments for Memory A:
{arguments_a}

## Arguments for Memory B:
{arguments_b}

## Original Memories:
Memory A: {memory_a_text}
Memory B: {memory_b_text}

Based on the dialectic arguments provided, synthesize a resolution that:
1. Acknowledges the contradiction
2. Weighs the evidence on each side
3. Provides a reasoned conclusion about which memory is more likely correct
4. Identifies what factors were most important in the decision

Return JSON with the following structure:
{{
  "resolution": "clear statement of the resolution",
  "winner": "A" or "B" or "inconclusive",
  "reasoning": "explanation of why this conclusion was reached",
  "confidence": 0.0-1.0,
  "recommendations": ["recommendation1", "recommendation2"]
}}

Return valid JSON only, no markdown or explanation.
"""

DIALECTIC_CHALLENGE_PROMPT = """
You are analyzing a challenge to a memory.

## Original Memory:
{memory_text}

## Challenge:
{challenge_text}

## Previous Challenges/Responses (if any):
{history}

Analyze the challenge and generate a response that:
1. Acknowledges the valid points in the challenge
2. Defends the memory if it is still considered accurate
3. Identifies any weaknesses in the memory that the challenge reveals
4. Determines if the memory should be adjusted, superseded, or maintained

Return JSON with the following structure:
{{
  "response": "your response to the challenge",
  "memory_status": "maintained" or "adjusted" or "superseded",
  "confidence_change": -0.3 to +0.3,
  "reasoning": "explanation of your decision"
}}

Return valid JSON only, no markdown or explanation.
"""


class DialecticSide(str, Enum):
    """Side of a dialectic argument."""

    PRO_A = "pro_a"
    PRO_B = "pro_b"


@dataclass
class DialecticArgument:
    """A single argument in a dialectic analysis."""

    memory_id: str
    side: str  # "pro_a" or "pro_b"
    argument: str
    confidence: float  # 0.0-1.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class DialecticResolution:
    """Result of dialectic resolution between two memories."""

    memory_a_id: str
    memory_b_id: str
    pro_arguments: list[DialecticArgument] = field(default_factory=list)
    con_arguments: list[DialecticArgument] = field(default_factory=list)
    resolution: str = ""
    winner: str = "inconclusive"  # "A", "B", "inconclusive"
    reasoning: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_a_id": self.memory_a_id,
            "memory_b_id": self.memory_b_id,
            "pro_arguments": [
                {
                    "memory_id": a.memory_id,
                    "side": a.side,
                    "argument": a.argument,
                    "confidence": a.confidence,
                    "evidence": a.evidence,
                }
                for a in self.pro_arguments
            ],
            "con_arguments": [
                {
                    "memory_id": a.memory_id,
                    "side": a.side,
                    "argument": a.argument,
                    "confidence": a.confidence,
                    "evidence": a.evidence,
                }
                for a in self.con_arguments
            ],
            "resolution": self.resolution,
            "winner": self.winner,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DialecticChallenge:
    """A challenge to a memory with response."""

    memory_id: str
    challenge_text: str
    response: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "challenge_text": self.challenge_text,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "confidence_delta": self.confidence_delta,
        }


@dataclass
class DialecticHistory:
    """History of dialectic activity for a memory."""

    memory_id: str
    notes: list[str] = field(default_factory=list)  # Dialectic reasoning notes
    challenges: list[DialecticChallenge] = field(default_factory=list)
    resolutions: list[DialecticResolution] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        """Add a dialectic note."""
        self.notes.append(note)

    def add_challenge(self, challenge: DialecticChallenge) -> None:
        """Add a challenge."""
        self.challenges.append(challenge)

    def add_resolution(self, resolution: DialecticResolution) -> None:
        """Add a resolution."""
        self.resolutions.append(resolution)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "notes": self.notes,
            "challenges": [c.to_dict() for c in self.challenges],
            "resolutions": [r.to_dict() for r in self.resolutions],
        }


# Storage key for dialectic history
DIALECTIC_TAG = "dialectic_history"


class DialecticEngine:
    """Dialectic reasoning engine for memory contradiction resolution.

    Uses LLM-based argumentation to build pro/con cases for conflicting memories
    and synthesize resolutions. Integrates with Memory Graph CONTRADICTS
    relationships and Trust Engine scoring.

    Features:
    - find_contradictions: Detect contradicting memory pairs
    - generate_arguments: Build pro/con cases using LLM
    - resolve_contradiction: Synthesize resolution from arguments
    - challenge_memory: Submit counter-arguments
    - get_dialectic_history: Retrieve dialectic activity

    All features opt-in via DIALECTIC_ENABLED config (default false).
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
    ) -> None:
        """Initialize DialecticEngine.

        Args:
            memory_system: Optional MemorySystem instance.
        """
        self._memory_system = memory_system
        self._config = get_config()
        self._enabled: bool | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._history_cache: dict[str, DialecticHistory] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if dialectic engine is enabled."""
        if self._enabled is None:
            self._enabled = self._config.dialectic_enabled
        return self._enabled

    @property
    def llm_provider(self) -> str:
        """Get LLM provider from config."""
        return self._config.dialectic_llm_provider

    @property
    def llm_model(self) -> str:
        """Get LLM model from config."""
        return self._config.dialectic_llm_model

    # =============================================================================
    # Contradiction Detection
    # =============================================================================

    async def find_contradictions(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find memory pairs that contradict each other.

        Args:
            query: Optional query string to find contradictions related to query.
            limit: Maximum number of contradiction pairs to return (default 10).

        Returns:
            List of dictionaries containing memory_a, memory_b pairs with IDs
            and text for each.
        """
        if not self.is_enabled:
            return [{"error": "Dialectic engine disabled"}]

        if self._memory_system is None:
            return [{"error": "Memory system not available"}]

        # Check if memory graph is enabled (required for CONTRADICTS relationships)
        if not self._config.memory_graph_enabled:
            logger.warning("dialectic_requires_memory_graph")
            return [{"error": "Memory graph disabled - cannot detect CONTRADICTS"}]

        try:
            # Get all memories
            all_memories = await self._memory_system.list_memories()
            contradictions: list[dict[str, Any]] = []

            # Find CONTRADICTS relationships
            for memory in all_memories:
                for rel in memory.relationships:
                    if rel.relationship_type == RelationshipType.CONTRADICTS:
                        # Found a contradiction relationship
                        # Get the target memory
                        target_memory = await self._memory_system.get_memory(
                            rel.target_id
                        )
                        if target_memory is None:
                            continue

                        # Filter by query if provided
                        if query:
                            query_lower = query.lower()
                            if (
                                query_lower not in memory.text.lower()
                                and query_lower not in target_memory.text.lower()
                            ):
                                continue

                        contradictions.append({
                            "memory_a": {
                                "id": memory.id,
                                "text": memory.text[:200],  # Truncate for display
                                "trust_score": memory.trust_score,
                            },
                            "memory_b": {
                                "id": target_memory.id,
                                "text": target_memory.text[:200],
                                "trust_score": target_memory.trust_score,
                            },
                            "confidence": rel.confidence,
                        })

                        if len(contradictions) >= limit:
                            break

                if len(contradictions) >= limit:
                    break

            return contradictions

        except Exception:
            logger.warning("dialectic_find_contradictions_failed", error=str(Exception))
            return [{"error": str(Exception)}]

    async def find_related_contradictions(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find contradictions related to a specific memory.

        Args:
            memory_id: ID of the memory to find contradictions for.
            limit: Maximum number of results (default 10).

        Returns:
            List of contradiction pairs involving this memory.
        """
        if not self.is_enabled:
            return [{"error": "Dialectic engine disabled"}]

        if self._memory_system is None:
            return [{"error": "Memory system not available"}]

        if not self._config.memory_graph_enabled:
            return [{"error": "Memory graph disabled - cannot detect CONTRADICTS"}]

        try:
            # Get the source memory
            source = await self._memory_system.get_memory(memory_id)
            if source is None:
                return [{"error": "Memory not found"}]

            contradictions: list[dict[str, Any]] = []

            # Find CONTRADICTS relationships
            for rel in source.relationships:
                if rel.relationship_type == RelationshipType.CONTRADICTS:
                    target = await self._memory_system.get_memory(rel.target_id)
                    if target:
                        contradictions.append({
                            "memory_a": {
                                "id": source.id,
                                "text": source.text[:200],
                                "trust_score": source.trust_score,
                            },
                            "memory_b": {
                                "id": target.id,
                                "text": target.text[:200],
                                "trust_score": target.trust_score,
                            },
                            "confidence": rel.confidence,
                        })

            # Also search reverse relationships (memories that contradict this one)
            all_memories = await self._memory_system.list_memories()
            for memory in all_memories:
                if memory.id == memory_id:
                    continue
                for rel in memory.relationships:
                    if (
                        rel.relationship_type == RelationshipType.CONTRADICTS
                        and rel.target_id == memory_id
                    ):
                        contradictions.append({
                            "memory_a": {
                                "id": memory.id,
                                "text": memory.text[:200],
                                "trust_score": memory.trust_score,
                            },
                            "memory_b": {
                                "id": source.id,
                                "text": source.text[:200],
                                "trust_score": source.trust_score,
                            },
                            "confidence": rel.confidence,
                        })

            return contradictions[:limit]

        except Exception:
            logger.warning("dialectic_find_related_contradictions_failed", error=str(Exception))
            return [{"error": str(Exception)}]

    # =============================================================================
    # Argument Generation
    # =============================================================================

    async def generate_arguments(
        self,
        memory_a_id: str,
        memory_b_id: str,
    ) -> dict[str, Any] | None:
        """Generate pro/con arguments for two contradictory memories.

        Args:
            memory_a_id: ID of the first memory.
            memory_b_id: ID of the second memory.

        Returns:
            Dictionary with dialectic analysis or None on failure.
        """
        if not self.is_enabled:
            return {"error": "Dialectic engine disabled"}

        if self._memory_system is None:
            return {"error": "Memory system not available"}

        try:
            # Get both memories
            memory_a = await self._memory_system.get_memory(memory_a_id)
            memory_b = await self._memory_system.get_memory(memory_b_id)

            if memory_a is None or memory_b is None:
                return {"error": "Memory not found"}

            # Generate arguments using LLM
            arguments_result = await self._call_llm(
                DIALECTIC_ARGUMENT_PROMPT.format(
                    memory_a_id=memory_a_id,
                    memory_a_text=memory_a.text,
                    memory_b_id=memory_b_id,
                    memory_b_text=memory_b.text,
                )
            )

            if arguments_result is None:
                return {"error": "LLM call failed"}

            # Parse the result
            try:
                # Look for JSON in the response
                json_match = re.search(
                    r'\{[^{}]*"pro_a_arguments"[^{}]*\}',
                    arguments_result,
                    re.DOTALL,
                )
                if not json_match:
                    # Try broader JSON match
                    json_match = re.search(r'\{.*\}', arguments_result, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return {"error": "Could not parse LLM response"}

            except json.JSONDecodeError:
                return {"error": "Invalid JSON from LLM"}

            # Build dialectic arguments
            pro_a_args = []
            for arg_data in data.get("pro_a_arguments", []):
                pro_a_args.append(
                    DialecticArgument(
                        memory_id=memory_a_id,
                        side=DialecticSide.PRO_A.value,
                        argument=arg_data.get("argument", ""),
                        confidence=arg_data.get("confidence", 0.5),
                        evidence=arg_data.get("evidence", []),
                    )
                )

            pro_b_args = []
            for arg_data in data.get("pro_b_arguments", []):
                pro_b_args.append(
                    DialecticArgument(
                        memory_id=memory_b_id,
                        side=DialecticSide.PRO_B.value,
                        argument=arg_data.get("argument", ""),
                        confidence=arg_data.get("confidence", 0.5),
                        evidence=arg_data.get("evidence", []),
                    )
                )

            # Return analysis
            return {
                "memory_a_id": memory_a_id,
                "memory_b_id": memory_b_id,
                "pro_a_arguments": [
                    {
                        "argument": a.argument,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in pro_a_args
                ],
                "pro_b_arguments": [
                    {
                        "argument": a.argument,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in pro_b_args
                ],
                "analysis": data.get("analysis", ""),
                "preferred_memory": data.get("preferred_memory", "neither"),
                "confidence": data.get("confidence", 0.5),
            }

        except Exception:
            logger.warning("dialectic_generate_arguments_failed", error=str(Exception))
            return {"error": str(Exception)}

    # =============================================================================
    # Resolution Synthesis
    # =============================================================================

    async def resolve_contradiction(
        self,
        memory_a_id: str,
        memory_b_id: str,
    ) -> DialecticResolution | None:
        """Generate dialectic resolution for two contradictory memories.

        Args:
            memory_a_id: ID of the first memory.
            memory_b_id: ID of the second memory.

        Returns:
            DialecticResolution object or None on failure.
        """
        if not self.is_enabled:
            return None

        if self._memory_system is None:
            return None

        try:
            # First generate arguments
            arguments = await self.generate_arguments(memory_a_id, memory_b_id)
            if arguments is None or "error" in arguments:
                return None

            # Get memories for resolution prompt
            memory_a = await self._memory_system.get_memory(memory_a_id)
            memory_b = await self._memory_system.get_memory(memory_b_id)

            if memory_a is None or memory_b is None:
                return None

            # Synthesize resolution
            resolution_text = await self._call_llm(
                DIALECTIC_RESOLUTION_PROMPT.format(
                    arguments_a=json.dumps(arguments.get("pro_a_arguments", [])),
                    arguments_b=json.dumps(arguments.get("pro_b_arguments", [])),
                    memory_a_text=memory_a.text,
                    memory_b_text=memory_b.text,
                )
            )

            if resolution_text is None:
                return None

            # Parse resolution
            try:
                json_match = re.search(r'\{.*\}', resolution_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return None

            except json.JSONDecodeError:
                return None

            # Build resolution
            pro_a_args = []
            for arg_data in arguments.get("pro_a_arguments", []):
                pro_a_args.append(
                    DialecticArgument(
                        memory_id=memory_a_id,
                        side=DialecticSide.PRO_A.value,
                        argument=arg_data.get("argument", ""),
                        confidence=arg_data.get("confidence", 0.5),
                        evidence=arg_data.get("evidence", []),
                    )
                )

            pro_b_args = []
            for arg_data in arguments.get("pro_b_arguments", []):
                pro_b_args.append(
                    DialecticArgument(
                        memory_id=memory_b_id,
                        side=DialecticSide.PRO_B.value,
                        argument=arg_data.get("argument", ""),
                        confidence=arg_data.get("confidence", 0.5),
                        evidence=arg_data.get("evidence", []),
                    )
                )

            resolution = DialecticResolution(
                memory_a_id=memory_a_id,
                memory_b_id=memory_b_id,
                pro_arguments=pro_a_args,
                con_arguments=pro_b_args,
                resolution=data.get("resolution", ""),
                winner=data.get("winner", "inconclusive"),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
            )

            # Store resolution in history
            await self._store_resolution(resolution)

            return resolution

        except Exception:
            logger.warning("dialectic_resolve_contradiction_failed", error=str(Exception))
            return None

    # =============================================================================
    # Challenge/Response
    # =============================================================================

    async def challenge_memory(
        self,
        memory_id: str,
        challenge_text: str,
    ) -> DialecticChallenge | None:
        """Submit a counter-argument challenge to a memory.

        Args:
            memory_id: ID of the memory to challenge.
            challenge_text: The challenge or counter-argument text.

        Returns:
            DialecticChallenge with response or None on failure.
        """
        if not self.is_enabled:
            return None

        if self._memory_system is None:
            return None

        try:
            # Get the memory
            memory = await self._memory_system.get_memory(memory_id)
            if memory is None:
                return None

            # Get dialectic history for context
            history = await self.get_dialectic_history(memory_id)
            history_str = ""
            if history and "challenges" in history:
                history_str = json.dumps(history["challenges"][-3:])  # Last 3

            # Generate response using LLM
            response_text = await self._call_llm(
                DIALECTIC_CHALLENGE_PROMPT.format(
                    memory_text=memory.text,
                    challenge_text=challenge_text,
                    history=history_str or "No previous challenges",
                )
            )

            if response_text is None:
                # Create challenge without response
                return DialecticChallenge(
                    memory_id=memory_id,
                    challenge_text=challenge_text,
                    response=None,
                    timestamp=datetime.utcnow(),
                )

            # Parse response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return DialecticChallenge(
                        memory_id=memory_id,
                        challenge_text=challenge_text,
                        response=None,
                        timestamp=datetime.utcnow(),
                    )

            except json.JSONDecodeError:
                return DialecticChallenge(
                    memory_id=memory_id,
                    challenge_text=challenge_text,
                    response=None,
                    timestamp=datetime.utcnow(),
                )

            # Build challenge
            challenge = DialecticChallenge(
                memory_id=memory_id,
                challenge_text=challenge_text,
                response=data.get("response"),
                timestamp=datetime.utcnow(),
                confidence_delta=data.get("confidence_change", 0.0),
            )

            # Apply confidence change if memory system supports trust
            if self._config.trust_engine_enabled and data.get("confidence_change"):
                await self._apply_confidence_delta(
                    memory_id, data.get("confidence_change", 0.0)
                )

            # Store challenge
            await self._store_challenge(challenge)

            return challenge

        except Exception:
            logger.warning("dialectic_challenge_memory_failed", error=str(Exception))
            return None

    # =============================================================================
    # History Management
    # =============================================================================

    async def get_dialectic_history(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:
        """Get dialectic history for a memory.

        Args:
            memory_id: ID of the memory.

        Returns:
            Dictionary with notes, challenges, and resolutions or None on failure.
        """
        if not self.is_enabled:
            return None

        # Check cache first
        if memory_id in self._history_cache:
            return self._history_cache[memory_id].to_dict()

        if self._memory_system is None:
            return None

        try:
            # Search for dialectic history in memory metadata
            from memini_ai.memory.schema import SearchFilter, SearchOptions

            filter_opts = SearchFilter()
            options = SearchOptions(topK=50, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"{DIALECTIC_TAG} {memory_id}", options
            )

            # Look for dialectic history entry
            history = DialecticHistory(memory_id=memory_id)

            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("dialectic_tag") == DIALECTIC_TAG:
                            # Found dialectic history
                            stored_history = json.loads(entry.text)
                            if stored_history.get("memory_id") == memory_id:
                                # Load notes
                                for note in stored_history.get("notes", []):
                                    history.add_note(note)
                                # Load challenges
                                for ch_data in stored_history.get("challenges", []):
                                    challenge = DialecticChallenge(
                                        memory_id=ch_data["memory_id"],
                                        challenge_text=ch_data["challenge_text"],
                                        response=ch_data.get("response"),
                                        timestamp=datetime.fromisoformat(
                                            ch_data["timestamp"]
                                        ),
                                        confidence_delta=ch_data.get(
                                            "confidence_delta", 0.0
                                        ),
                                    )
                                    history.add_challenge(challenge)
                                # Load resolutions
                                for res_data in stored_history.get("resolutions", []):
                                    res = DialecticResolution(
                                        memory_a_id=res_data["memory_a_id"],
                                        memory_b_id=res_data["memory_b_id"],
                                        resolution=res_data["resolution"],
                                        winner=res_data.get("winner", "inconclusive"),
                                        reasoning=res_data.get("reasoning", ""),
                                        confidence=res_data.get("confidence", 0.5),
                                        timestamp=datetime.fromisoformat(
                                            res_data["timestamp"]
                                        ),
                                    )
                                    history.add_resolution(res)

                                break
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Cache the result
            self._history_cache[memory_id] = history
            return history.to_dict()

        except Exception:
            logger.warning("dialectic_get_history_failed", error=str(Exception))
            return None

    async def _store_resolution(self, resolution: DialecticResolution) -> None:
        """Store a dialectic resolution in memory."""
        if self._memory_system is None:
            return

        # Update cache
        memory_id = resolution.memory_a_id
        if memory_id not in self._history_cache:
            self._history_cache[memory_id] = DialecticHistory(memory_id=memory_id)
        self._history_cache[memory_id].add_resolution(resolution)

        # Also store for memory_b
        memory_id_b = resolution.memory_b_id
        if memory_id_b not in self._history_cache:
            self._history_cache[memory_id_b] = DialecticHistory(memory_id=memory_id_b)

        # Persist to storage
        await self._persist_history(resolution.memory_a_id)
        await self._persist_history(resolution.memory_b_id)

    async def _store_challenge(self, challenge: DialecticChallenge) -> None:
        """Store a dialectic challenge in memory."""
        # Update cache
        if challenge.memory_id not in self._history_cache:
            self._history_cache[challenge.memory_id] = DialecticHistory(
                memory_id=challenge.memory_id
            )
        self._history_cache[challenge.memory_id].add_challenge(challenge)

        # Persist
        await self._persist_history(challenge.memory_id)

    async def _persist_history(self, memory_id: str) -> None:
        """Persist dialectic history to memory storage."""
        if self._memory_system is None:
            return

        if memory_id not in self._history_cache:
            return

        try:
            from memini_ai.memory.schema import MemoryEntry, MemorySourceType

            history = self._history_cache[memory_id]
            history_json = json.dumps(history.to_dict())

            metadata = {
                "dialectic_tag": DIALECTIC_TAG,
                "memory_id": memory_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            entry = MemoryEntry(
                text=history_json,
                sourceType=MemorySourceType.project,
                metadataJson=json.dumps(metadata),
            )

            # Find and update existing or create new
            existing_id = await self._find_dialectic_memory_id(memory_id)
            if existing_id:
                await self._memory_system.delete_memory(existing_id)

            await self._memory_system.add_memory(entry)

        except Exception:
            logger.warning("dialectic_persist_history_failed", error=str(Exception))

    async def _find_dialectic_memory_id(self, memory_id: str) -> str | None:
        """Find existing dialectic history memory ID."""
        if self._memory_system is None:
            return None

        try:
            from memini_ai.memory.schema import SearchFilter, SearchOptions

            filter_opts = SearchFilter()
            options = SearchOptions(topK=20, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"{DIALECTIC_TAG} {memory_id}", options
            )

            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if (
                            metadata.get("dialectic_tag") == DIALECTIC_TAG
                            and metadata.get("memory_id") == memory_id
                        ):
                            return entry.id
                    except json.JSONDecodeError:
                        continue

        except Exception:
            pass

        return None

    async def _apply_confidence_delta(
        self,
        memory_id: str,
        delta: float,
    ) -> None:
        """Apply confidence delta to a memory (via Trust Engine)."""
        if self._memory_system is None:
            return

        try:
            from memini_ai.trust_engine import TrustEngine

            trust_engine = TrustEngine(memory_system=self._memory_system)

            # Map delta to trust signal
            if delta > 0:
                signal = "agent_used"
            else:
                signal = "agent_ignored"

            await trust_engine.adjust_trust(memory_id, signal)

        except Exception:
            # Silently fail - trust engine might not be enabled
            pass

    # =============================================================================
    # LLM Communication
    # =============================================================================

    async def _call_llm(self, prompt: str) -> str | None:
        """Call LLM with dialectic prompt.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            LLM response text or None on failure.
        """
        try:
            client = await self._get_http_client()

            # Build request based on provider
            if self.llm_provider == "ollama":
                response = await client.post(
                    self._config.llm_url,
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
            elif self.llm_provider in ("openai", "anthropic"):
                # Use compatible format for both
                response = await client.post(
                    self._config.llm_url,
                    json={
                        "model": self.llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
            else:
                # Default to ollama format
                response = await client.post(
                    self._config.llm_url,
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )

            if response.status_code == 200:
                result = response.json()
                return str(result.get("response", ""))

        except Exception:
            logger.warning("dialectic_llm_call_failed", error=str(Exception))

        return None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for LLM calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Module-level singleton
_dialectic_engine: DialecticEngine | None = None


def get_dialectic_engine(memory_system: MemorySystem | None = None) -> DialecticEngine:
    """Get or create the global DialecticEngine instance.

    Args:
        memory_system: Optional MemorySystem to use.

    Returns:
        DialecticEngine instance.
    """
    global _dialectic_engine
    if _dialectic_engine is None:
        _dialectic_engine = DialecticEngine(memory_system=memory_system)
    return _dialectic_engine
