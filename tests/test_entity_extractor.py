"""Tests for entity_extractor.py - Phase 4B Entity Extractor."""

import pytest

from memini_ai.entity_extractor import EntityExtractor
from memini_ai.knowledge_graph import EntityType


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        """Create an entity extractor instance."""
        return EntityExtractor()

    def test_extractor_creation(self, extractor: EntityExtractor) -> None:
        """Test extractor initializes correctly."""
        assert extractor is not None
        assert len(extractor._compiled_patterns) > 0

    def test_extract_file_extensions(self, extractor: EntityExtractor) -> None:
        """Test extracting file extensions."""
        text = "The config file is at /path/to/config.yaml and settings.json"
        entities = extractor.extract_from_text(text)
        assert len(entities) > 0
        names = [e["name"] for e in entities]
        assert any("config.yaml" in n or "settings.json" in n for n in names)

    def test_extract_function_calls(self, extractor: EntityExtractor) -> None:
        """Test extracting function calls."""
        text = "Call process_data() to handle the input and validate() for checks."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("process_data" in n for n in names)
        assert any("validate" in n for n in names)

    def test_extract_function_definitions(self, extractor: EntityExtractor) -> None:
        """Test extracting function definitions."""
        text = "def calculate_total(items): return sum(items)"
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("calculate_total" in n for n in names)

    def test_extract_class_names(self, extractor: EntityExtractor) -> None:
        """Test extracting class names."""
        text = "The UserService handles authentication and the OrderProcessor processes orders."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("UserService" in n for n in names)
        assert any("OrderProcessor" in n for n in names)

    def test_extract_constants(self, extractor: EntityExtractor) -> None:
        """Test extracting constants."""
        text = "MAX_CONNECTIONS and DEFAULT_TIMEOUT are configuration values."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("MAX_CONNECTIONS" in n for n in names)
        assert any("DEFAULT_TIMEOUT" in n for n in names)

    def test_extract_urls(self, extractor: EntityExtractor) -> None:
        """Test extracting URLs."""
        text = "Visit https://example.com/api or http://localhost:8080 for details."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("https://example.com/api" in n for n in names)
        assert any("http://localhost:8080" in n for n in names)

    def test_extract_person_names(self, extractor: EntityExtractor) -> None:
        """Test extracting person names."""
        text = "John Smith and Jane Doe contributed to this project."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("John Smith" in n for n in names)
        assert any("Jane Doe" in n for n in names)

    def test_extract_repo_paths(self, extractor: EntityExtractor) -> None:
        """Test extracting repo paths."""
        text = "The code is at owner/repository and another at team/project."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("owner/repository" in n for n in names)
        assert any("team/project" in n for n in names)

    def test_extract_table_names(self, extractor: EntityExtractor) -> None:
        """Test extracting database table names."""
        text = "Query the user_accounts table and the order_items table."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("user_accounts" in n for n in names)
        assert any("order_items" in n for n in names)

    def test_extract_env_variables(self, extractor: EntityExtractor) -> None:
        """Test extracting environment variables."""
        text = "Set $DATABASE_URL and $API_KEY in the environment."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("DATABASE_URL" in n for n in names)
        assert any("API_KEY" in n for n in names)

    def test_extract_quoted_strings(self, extractor: EntityExtractor) -> None:
        """Test extracting quoted strings."""
        text = 'The "Memory System" component uses the "Graph Database".'
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("Memory System" in n for n in names)
        assert any("Graph Database" in n for n in names)

    def test_confidence_scores(self, extractor: EntityExtractor) -> None:
        """Test confidence scores are calculated."""
        text = "Call process_data() on UserService"
        entities = extractor.extract_from_text(text)
        for entity in entities:
            assert 0.0 <= entity["confidence"] <= 1.0

    def test_entity_types_assigned(self, extractor: EntityExtractor) -> None:
        """Test entity types are assigned correctly."""
        text = "def my_function(): pass"
        entities = extractor.extract_from_text(text)
        assert len(entities) > 0
        # Function definitions should be CODE type
        code_entities = [e for e in entities if e["type"] == EntityType.CODE]
        assert len(code_entities) > 0

    def test_deduplication(self, extractor: EntityExtractor) -> None:
        """Test deduplication of entities."""
        text = (
            "The UserService is used by UserService to handle UserService operations."
        )
        entities = extractor.extract_from_text(text)
        names_lower = [e["name"].lower() for e in entities]
        # Should not have duplicates
        assert len(names_lower) == len(set(names_lower))

    def test_false_positive_filtering(self, extractor: EntityExtractor) -> None:
        """Test false positives are filtered."""
        text = "True False None TODO FIXME NOTE DEBUG"
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        # Should not include these as entities
        for fp in ["True", "False", "None", "TODO"]:
            assert fp not in names

    def test_stop_word_filtering(self, extractor: EntityExtractor) -> None:
        """Test stop words are filtered."""
        text = "The and or but in on at to for"
        entities = extractor.extract_from_text(text)
        names_lower = [e["name"].lower() for e in entities]
        # Should not include stop words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for"}
        for sw in stop_words:
            assert sw not in names_lower

    def test_extract_with_context(self, extractor: EntityExtractor) -> None:
        """Test extraction with context."""
        text = "The function process_data handles input processing."
        entities = extractor.extract_with_context(text, context_window=10)
        assert len(entities) > 0
        # Should have context field
        for entity in entities:
            assert "context" in entity

    def test_resolve_canonical_form_snake_case(
        self, extractor: EntityExtractor
    ) -> None:
        """Test canonical form for snake_case."""
        canonical = extractor.resolve_canonical_form("my_function_name")
        assert canonical == "my_function_name"

    def test_resolve_canonical_form_pascal_case(
        self, extractor: EntityExtractor
    ) -> None:
        """Test canonical form for PascalCase."""
        canonical = extractor.resolve_canonical_form("MyClassName")
        assert canonical == "MyClassName"

    def test_resolve_canonical_form_title_case(
        self, extractor: EntityExtractor
    ) -> None:
        """Test canonical form for title case phrases."""
        canonical = extractor.resolve_canonical_form("memory system")
        assert canonical == "Memory System"

    def test_filter_by_type(self, extractor: EntityExtractor) -> None:
        """Test filtering entities by type."""
        text = "def my_function(): pass"
        entities = extractor.extract_from_text(text)
        code_entities = extractor.filter_by_type(entities, EntityType.CODE)
        assert all(e["type"] == EntityType.CODE for e in code_entities)

    def test_get_type_distribution(self, extractor: EntityExtractor) -> None:
        """Test getting type distribution."""
        text = "def my_function(): pass"
        entities = extractor.extract_from_text(text)
        distribution = extractor.get_type_distribution(entities)
        assert isinstance(distribution, dict)
        assert all(isinstance(v, int) for v in distribution.values())

    def test_deduplicate_by_similarity(self, extractor: EntityExtractor) -> None:
        """Test deduplication by similarity."""
        entities = [
            {
                "name": "MyFunction",
                "type": EntityType.CODE,
                "confidence": 0.9,
                "pattern_matched": "",
            },
            {
                "name": "myfunction",
                "type": EntityType.CODE,
                "confidence": 0.8,
                "pattern_matched": "",
            },
            {
                "name": "OtherFunction",
                "type": EntityType.CODE,
                "confidence": 0.7,
                "pattern_matched": "",
            },
        ]
        deduped = extractor.deduplicate_by_similarity(entities)
        assert len(deduped) == 2  # MyFunction and myfunction should merge

    def test_minimum_length_filter(self, extractor: EntityExtractor) -> None:
        """Test entities below minimum length are filtered."""
        text = "a b cd ef"
        entities = extractor.extract_from_text(text)
        # Single chars should be filtered
        for entity in entities:
            assert len(entity["name"]) >= 2

    def test_empty_text(self, extractor: EntityExtractor) -> None:
        """Test extracting from empty text."""
        entities = extractor.extract_from_text("")
        assert entities == []

    def test_code_entity_boost(self, extractor: EntityExtractor) -> None:
        """Test code entities get confidence boost."""
        text = "my_function"
        entities = extractor.extract_from_text(text)
        if entities:
            # snake_case should get a confidence boost
            assert entities[0]["confidence"] > 0.7

    def test_long_name_boost(self, extractor: EntityExtractor) -> None:
        """Test longer names get confidence boost."""
        text = "ThisFunctionHasALongName"
        entities = extractor.extract_from_text(text)
        if entities:
            # Long names should have higher confidence
            assert entities[0]["confidence"] > 0.7


class TestEntityExtractorPatterns:
    """Test specific regex patterns."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_camel_case_detection(self, extractor: EntityExtractor) -> None:
        """Test camelCase variable detection."""
        text = "The userName variable holds the login."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("userName" in n for n in names)

    def test_snake_case_detection(self, extractor: EntityExtractor) -> None:
        """Test snake_case function detection."""
        text = "Use get_user_data() to fetch records."
        entities = extractor.extract_from_text(text)
        names = [e["name"] for e in entities]
        assert any("get_user_data" in n for n in names)

    def test_mixed_content(self, extractor: EntityExtractor) -> None:
        """Test extraction from mixed content."""
        text = """
        The UserService class uses process_user_data() to handle authentication.
        It connects to DATABASE_URL and stores logs in /var/log/app.log.
        Visit https://api.example.com for more info.
        John Doe and Jane Smith are the maintainers.
        """
        entities = extractor.extract_from_text(text)
        assert len(entities) > 0

        # Should have various types
        types = {e["type"] for e in entities}
        assert len(types) > 1  # At least CODE and something else
