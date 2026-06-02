"""User Modeling - Persistent user profiles with dialectic LLM reasoning."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem

from memini_ai.config import get_config
from memini_ai.llm.factory import get_llm_client
from memini_ai.memory.schema import MemorySourceType, UserProfile
from memini_ai.utils.logger import logger

# Profile storage key for memory entry
PROFILE_MEMORY_TAG = "user_profile"
PROFILE_COLLECTION_NAME = "user_profiles"


# Dialectic reasoning prompt for profile updates
DIALECTIC_UPDATE_PROMPT = """
You are analyzing conversation data to build a user profile.

Given the following conversation summary, infer and update the user profile:

## Previous Profile Understanding:
{profile_summary}

## Recent Conversation:
{conversation}

Identify and update:

1. **Communication Style**: Does the user prefer concise or detailed responses?
   Technical or plain language? Formal or casual?

2. **Expertise Level**: What domains does the user demonstrate expertise in?
   What topics are they learning or unfamiliar with?

3. **Preferences**: What specific preferences can you infer?
   (Response format, code style, workflow preferences, etc.)

4. **Confidence**: How confident are you in this profile?
   (Based on consistency across sessions)

Return JSON with the following structure:
{{
  "communication_style": "concise|detailed|technical|plain|neutral",
  "expertise_domains": ["domain1", "domain2"],
  "preferences": {{"key": "value"}},
  "confidence": 0.0-1.0,
  "reasoning": "your reasoning trace"
}}

Return valid JSON only, no markdown or explanation.
"""

# Profile query prompt for brief summary
GET_PROFILE_PROMPT = """
Given the current user profile and recent context, generate a brief profile
summary for context injection (under 200 tokens).

Focus on what's most relevant for tailoring responses.

Profile:
{profile}

Summary:
"""

# New profile initialization prompt
INIT_PROFILE_PROMPT = """
Analyze this initial conversation to create a user profile.

Conversation:
{conversation}

Identify:

1. **Communication Style**: Does the user prefer concise or detailed responses?
   Technical or plain language?

2. **Expertise Level**: What domains does the user demonstrate expertise in?

3. **Any Stated Preferences**: What has the user explicitly requested or preferred?

Return JSON:
{{
  "communication_style": "concise|detailed|technical|plain|neutral",
  "expertise_domains": ["domain1"],
  "preferences": {{}},
  "confidence": 0.3,
  "reasoning": "initial reasoning"
}}

Return valid JSON only.
"""


class UserModel:
    """Dialectic user modeling with LLM-based profile updates.

    Features:
    - Persistent user profile stored as memory entries
    - Dialectic LLM reasoning for profile updates
    - Communication style tracking
    - Expertise domain tracking
    - Preference inference
    - Requires 50-100 sessions before reliable
    - Optional (USER_MODELING_ENABLED=false disables)

    Profile updates happen after each session using LLM reasoning
    over the conversation history and previous profile.
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
    ) -> None:
        self._memory_system = memory_system
        self._config = get_config()
        self._enabled: bool | None = None
        self._profile_cache: UserProfile | None = None
        self._http_client = None

    async def close(self) -> None:
        """Cleanup resources."""
        if self._http_client is not None and hasattr(self._http_client, "aclose"):
            await self._http_client.aclose()
        self._http_client = None

    @property
    def is_enabled(self) -> bool:
        """Check if user modeling is enabled."""
        if self._enabled is None:
            self._enabled = self._config.user_modeling_enabled
        return self._enabled

    @property
    def is_warmed_up(self) -> bool:
        """Check if we have enough sessions for reliable profile."""
        if self._profile_cache is None:
            return False
        return self._profile_cache.session_count >= self._config.user_model_min_sessions

    @property
    def session_count(self) -> int:
        """Get current session count from cached profile."""
        if self._profile_cache is None:
            return 0
        return self._profile_cache.session_count

    async def get_profile(
        self,
        include_dialectic_notes: bool = False,
    ) -> dict[str, Any]:
        """Get current user profile.

        Args:
            include_dialectic_notes: Include LLM reasoning traces.

        Returns:
            Dictionary with profile data.
        """
        if not self.is_enabled:
            return {"error": "User modeling disabled"}

        # Load profile from memory
        profile = await self._load_profile()
        if profile is None:
            return {
                "error": "No profile found",
                "warmed_up": False,
                "session_count": 0,
            }

        result = {
            "user_id": profile.user_id,
            "communication_style": profile.communication_style,
            "expertise_domains": profile.expertise_domains,
            "preferences": profile.preferences,
            "confidence": profile.confidence,
            "last_updated": profile.last_updated.isoformat(),
            "session_count": profile.session_count,
            "warmed_up": self.is_warmed_up,
        }

        if include_dialectic_notes:
            result["dialectic_notes"] = profile.dialectic_notes[-5:]  # Last 5

        return result

    async def get_profile_summary(self) -> str | None:
        """Get brief profile summary for context injection.

        Returns:
            Brief profile summary string or None if disabled/unavailable.
        """
        if not self.is_enabled:
            return None

        profile = await self._load_profile()
        if profile is None:
            return None

        # If not warmed up yet, return None (profile not reliable)
        if not self.is_warmed_up:
            return None

        try:
            client = get_llm_client(self._config)
            profile_json = json.dumps(
                {
                    "style": profile.communication_style,
                    "domains": profile.expertise_domains,
                    "preferences": profile.preferences,
                }
            )

            text = await client.generate(
                prompt=GET_PROFILE_PROMPT.format(profile=profile_json),
            )

            if text:
                return text.strip()
        except Exception:
            logger.warning("user_model_profile_summary_failed", error=str(Exception))

        return None

    async def update_profile_from_session(
        self,
        conversation: str,
    ) -> dict[str, Any]:
        """Update user profile dialectically after a session.

        Args:
            conversation: Conversation text from the session.

        Returns:
            Dictionary with update status and reasoning.
        """
        if not self.is_enabled:
            return {"success": False, "error": "User modeling disabled"}

        if not conversation or not conversation.strip():
            return {"success": False, "error": "Empty conversation"}

        # Load current profile
        current_profile = await self._load_profile()

        # Create default if not exists
        if current_profile is None:
            current_profile = UserProfile(
                user_id=self._config.effective_project_id,
                session_count=0,
            )

        # Perform dialectic reasoning
        reasoning = await self._dialectic_update(current_profile, conversation)

        if reasoning:
            # Update profile
            current_profile.session_count += 1
            current_profile.last_updated = datetime.utcnow()
            current_profile.dialectic_notes.append(reasoning)

            # Save updated profile
            await self._save_profile(current_profile)
            self._profile_cache = current_profile

            return {
                "success": True,
                "session_count": current_profile.session_count,
                "reasoning": reasoning,
                "warmed_up": self.is_warmed_up,
            }

        return {"success": False, "error": "Dialectic reasoning failed"}

    async def _dialectic_update(
        self,
        profile: UserProfile,
        conversation: str,
    ) -> str | None:
        """Perform dialectic reasoning to update profile.

        Args:
            profile: Current user profile.
            conversation: Conversation text.

        Returns:
            Reasoning text from LLM, or None on failure.
        """
        try:
            client = get_llm_client(self._config)

            # Build profile summary for prompt
            profile_summary = json.dumps(
                {
                    "communication_style": profile.communication_style,
                    "expertise_domains": profile.expertise_domains,
                    "preferences": profile.preferences,
                    "confidence": profile.confidence,
                    "session_count": profile.session_count,
                }
            )

            # Choose prompt based on whether this is a new profile
            if profile.session_count == 0:
                prompt_template = INIT_PROFILE_PROMPT
            else:
                prompt_template = DIALECTIC_UPDATE_PROMPT

            reasoning = await client.generate(
                prompt=prompt_template.format(
                    profile_summary=profile_summary,
                    conversation=conversation,
                ),
            )

            if reasoning:
                # Parse and apply updates
                await self._apply_profile_update(profile, reasoning)

                return reasoning[:1000]  # Truncate for storage

        except Exception:
            logger.warning("user_model_dialectic_update_failed", error=str(Exception))

        return None

    async def _apply_profile_update(
        self,
        profile: UserProfile,
        reasoning: str,
    ) -> None:
        """Parse LLM reasoning and update profile fields.

        Args:
            profile: Profile to update.
            reasoning: LLM reasoning text.
        """
        # Try to parse JSON from reasoning
        try:
            # Look for JSON object in reasoning
            json_match = re.search(r'\{[^{}]*"[^{}]*\}', reasoning, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Update communication style
                if "communication_style" in data:
                    style = data["communication_style"].lower()
                    if style in (
                        "concise",
                        "detailed",
                        "technical",
                        "plain",
                        "neutral",
                    ):
                        profile.communication_style = style

                # Update expertise domains
                if "expertise_domains" in data and isinstance(
                    data["expertise_domains"], list
                ):
                    profile.expertise_domains = data["expertise_domains"]

                # Update preferences
                if "preferences" in data and isinstance(data["preferences"], dict):
                    profile.preferences.update(data["preferences"])

                # Update confidence
                if "confidence" in data and isinstance(
                    data["confidence"], (int, float)
                ):
                    profile.confidence = max(0.0, min(1.0, float(data["confidence"])))

        except (json.JSONDecodeError, AttributeError, ValueError):
            # If parsing fails, just keep existing profile
            # Confidence stays the same or decreases slightly on failed update
            profile.confidence = max(0.0, profile.confidence * 0.95)

    async def _load_profile(self) -> UserProfile | None:
        """Load profile from memory storage.

        Returns:
            UserProfile if found, None otherwise.
        """
        # Return cached if available
        if self._profile_cache is not None:
            return self._profile_cache

        if self._memory_system is None:
            return None

        # Search for profile memory entry
        try:
            from memini_ai.memory.schema import SearchFilter, SearchOptions

            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=10, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"user_profile {self._config.effective_project_id}", options
            )

            # Look for profile entry in results
            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("profile_tag") == PROFILE_MEMORY_TAG:
                            profile_data = json.loads(entry.text)
                            profile = UserProfile.from_dict(profile_data)
                            self._profile_cache = profile
                            return profile
                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception:
            logger.warning("user_model_load_profile_failed", error=str(Exception))

        return None

    async def _save_profile(self, profile: UserProfile) -> None:
        """Save profile to memory storage.

        Args:
            profile: UserProfile to save.
        """
        if self._memory_system is None:
            return

        try:
            from memini_ai.memory.schema import MemoryEntry

            # Serialize profile
            profile_json = json.dumps(profile.to_dict())

            # Create metadata
            metadata = {
                "profile_tag": PROFILE_MEMORY_TAG,
                "user_id": profile.user_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Create memory entry for profile
            entry = MemoryEntry(
                text=profile_json,
                sourceType=MemorySourceType.project,
                metadataJson=json.dumps(metadata),
            )

            # Try to find and update existing profile entry
            existing_id = await self._find_profile_memory_id()
            if existing_id:
                # Delete old entry
                await self._memory_system.delete_memory(existing_id)

            # Add new entry
            memory_id = await self._memory_system.add_memory(entry)
            logger.info("user_model_profile_saved", memory_id=memory_id)

        except Exception:
            logger.warning("user_model_save_profile_failed", error=str(Exception))

    async def _find_profile_memory_id(self) -> str | None:
        """Find the memory ID of the current user's profile.

        Returns:
            Memory ID if found, None otherwise.
        """
        if self._memory_system is None:
            return None

        try:
            from memini_ai.memory.schema import SearchFilter, SearchOptions

            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=20, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"user_profile {self._config.effective_project_id}", options
            )

            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("profile_tag") == PROFILE_MEMORY_TAG:
                            return entry.id
                    except json.JSONDecodeError:
                        continue

        except Exception:
            pass

        return None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for LLM calls (legacy stub)."""
        raise NotImplementedError(
            "HTTP client creation moved to LLM factory. This is a dead method stub."
        )


# Module-level singleton
_user_model: UserModel | None = None


def get_user_model(memory_system: MemorySystem | None = None) -> UserModel:
    """Get or create the global UserModel instance.

    Args:
        memory_system: Optional MemorySystem to use.

    Returns:
        UserModel instance.
    """
    global _user_model
    if _user_model is None:
        _user_model = UserModel(memory_system=memory_system)
    return _user_model
