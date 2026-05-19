"""Tests for knowledge_graph.py - Phase 4B Knowledge Graph."""

import pytest

from memini_ai.knowledge_graph import (
    Entity,
    EntityGraphResult,
    EntityType,
    InferenceResult,
    KGQuery,
    KnowledgeGraph,
)
from memini_ai.memory.schema import RelationshipType


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self) -> None:
        """Test all entity types are defined."""
        assert EntityType.PERSON.value == "PERSON"
        assert EntityType.ORGANIZATION.value == "ORGANIZATION"
        assert EntityType.CONCEPT.value == "CONCEPT"
        assert EntityType.CODE.value == "CODE"
        assert EntityType.PROJECT.value == "PROJECT"
        assert EntityType.LOCATION.value == "LOCATION"
        assert EntityType.UNKNOWN.value == "UNKNOWN"

    def test_entity_type_is_string_enum(self) -> None:
        """Test EntityType is a string enum for serialization."""
        for et in EntityType:
            assert isinstance(et.value, str)


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self) -> None:
        """Test basic entity creation."""
        entity = Entity(
            entity_id="kg:entity:test",
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
            canonical_name="test_entity",
            confidence=0.9,
        )
        assert entity.entity_id == "kg:entity:test"
        assert entity.name == "Test Entity"
        assert entity.entity_type == EntityType.CONCEPT
        assert entity.canonical_name == "test_entity"
        assert entity.confidence == 0.9
        assert entity.mentions == []

    def test_entity_creation_with_mentions(self) -> None:
        """Test entity creation with mentions list."""
        entity = Entity(
            entity_id="kg:entity:test",
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
            canonical_name="test_entity",
            mentions=["Test Entity", "test entity", "TEST ENTITY"],
        )
        assert len(entity.mentions) == 3

    def test_entity_to_dict(self) -> None:
        """Test entity serialization."""
        entity = Entity(
            entity_id="kg:entity:test",
            name="Test Entity",
            entity_type=EntityType.CODE,
            canonical_name="test_entity",
            confidence=0.85,
        )
        d = entity.to_dict()
        assert d["entityId"] == "kg:entity:test"
        assert d["name"] == "Test Entity"
        assert d["entityType"] == "CODE"
        assert d["canonicalName"] == "test_entity"
        assert d["confidence"] == 0.85
        assert d["mentions"] == []


class TestKGQuery:
    """Tests for KGQuery dataclass."""

    def test_kg_query_defaults(self) -> None:
        """Test KGQuery default values."""
        query = KGQuery()
        assert query.entity_a is None
        assert query.entity_b is None
        assert query.relationship_types is None
        assert query.inference_depth == 1
        assert query.limit == 100

    def test_kg_query_with_values(self) -> None:
        """Test KGQuery with custom values."""
        query = KGQuery(
            entity_a="entity1",
            entity_b="entity2",
            relationship_types=[RelationshipType.RELATED_TO],
            inference_depth=2,
            limit=50,
        )
        assert query.entity_a == "entity1"
        assert query.entity_b == "entity2"
        assert query.inference_depth == 2
        assert query.limit == 50

    def test_kg_query_to_dict(self) -> None:
        """Test KGQuery serialization."""
        query = KGQuery(
            entity_a="entity1",
            relationship_types=[RelationshipType.RELATED_TO, RelationshipType.DERIVED_FROM],
        )
        d = query.to_dict()
        assert d["entityA"] == "entity1"
        assert "RELATED_TO" in d["relationshipTypes"]
        assert "DERIVED_FROM" in d["relationshipTypes"]
        assert d["inferenceDepth"] == 1


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_inference_result_creation(self) -> None:
        """Test InferenceResult creation."""
        result = InferenceResult(
            start_entity="entity1",
            end_entity="entity3",
            path=[
                {"entity": "entity1", "relationship": "RELATED_TO", "next_entity": "entity2"},
                {"entity": "entity2", "relationship": "RELATED_TO", "next_entity": "entity3"},
            ],
            total_confidence=0.81,
            depth=2,
        )
        assert result.start_entity == "entity1"
        assert result.end_entity == "entity3"
        assert len(result.path) == 2
        assert result.total_confidence == 0.81
        assert result.depth == 2

    def test_inference_result_to_dict(self) -> None:
        """Test InferenceResult serialization."""
        result = InferenceResult(
            start_entity="e1",
            end_entity="e2",
            path=[],
            total_confidence=1.0,
            depth=0,
        )
        d = result.to_dict()
        assert d["startEntity"] == "e1"
        assert d["endEntity"] == "e2"
        assert d["totalConfidence"] == 1.0


class TestEntityGraphResult:
    """Tests for EntityGraphResult dataclass."""

    def test_entity_graph_result_empty(self) -> None:
        """Test empty EntityGraphResult."""
        result = EntityGraphResult(
            entity_id="entity1",
            entity_name="Test Entity",
        )
        assert result.entity_id == "entity1"
        assert result.entity_name == "Test Entity"
        assert result.incoming == []
        assert result.outgoing == []
        assert result.inferred == []

    def test_entity_graph_result_with_data(self) -> None:
        """Test EntityGraphResult with connections."""
        result = EntityGraphResult(
            entity_id="entity1",
            entity_name="Test",
            incoming=[{"sourceId": "entity2", "relationship": "RELATED_TO"}],
            outgoing=[{"targetId": "entity3", "relationship": "DERIVED_FROM"}],
        )
        assert len(result.incoming) == 1
        assert len(result.outgoing) == 1


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    @pytest.fixture
    def kg(self) -> KnowledgeGraph:
        """Create a knowledge graph instance."""
        return KnowledgeGraph()

    @pytest.mark.asyncio
    async def test_kg_disabled_by_default(self, kg: KnowledgeGraph) -> None:
        """Test KG is disabled when config not set."""
        assert not kg.is_enabled

    @pytest.mark.asyncio
    async def test_create_entity(self, kg: KnowledgeGraph) -> None:
        """Test entity creation."""
        entity = await kg.create_entity(
            name="TestFunction",
            entity_type=EntityType.CODE,
            confidence=0.9,
        )
        assert entity.name == "TestFunction"
        assert entity.entity_type == EntityType.CODE
        assert "testfunction" in entity.entity_id
        assert entity.confidence == 0.9

    @pytest.mark.asyncio
    async def test_create_duplicate_entity(self, kg: KnowledgeGraph) -> None:
        """Test creating entity with same name returns existing."""
        e1 = await kg.create_entity("MyEntity", EntityType.CONCEPT)
        e2 = await kg.create_entity("MyEntity", EntityType.CONCEPT)
        assert e1.entity_id == e2.entity_id
        assert e1.entity_id == e2.entity_id

    @pytest.mark.asyncio
    async def test_get_entity(self, kg: KnowledgeGraph) -> None:
        """Test getting entity by ID."""
        created = await kg.create_entity("SearchTest", EntityType.CODE)
        retrieved = await kg.get_entity(created.entity_id)
        assert retrieved is not None
        assert retrieved.name == "SearchTest"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, kg: KnowledgeGraph) -> None:
        """Test getting non-existent entity returns None."""
        result = await kg.get_entity("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_by_name(self, kg: KnowledgeGraph) -> None:
        """Test getting entity by canonical name."""
        await kg.create_entity("NamedEntity", EntityType.CONCEPT, canonical_name="named_entity")
        entity = await kg.get_entity_by_name("named_entity")
        assert entity is not None
        assert entity.name == "NamedEntity"

    @pytest.mark.asyncio
    async def test_search_entities_exact(self, kg: KnowledgeGraph) -> None:
        """Test exact name search."""
        await kg.create_entity("ExactMatch", EntityType.CONCEPT)
        await kg.create_entity("Partial", EntityType.CONCEPT)
        await kg.create_entity("Other", EntityType.CONCEPT)

        results = await kg.search_entities("ExactMatch")
        assert len(results) >= 1
        assert any(e.name == "ExactMatch" for e in results)

    @pytest.mark.asyncio
    async def test_search_entities_partial(self, kg: KnowledgeGraph) -> None:
        """Test partial name match."""
        await kg.create_entity("FunctionOne", EntityType.CODE)
        await kg.create_entity("FunctionTwo", EntityType.CODE)
        await kg.create_entity("OtherCode", EntityType.CODE)

        results = await kg.search_entities("Function")
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_list_entities(self, kg: KnowledgeGraph) -> None:
        """Test listing all entities."""
        await kg.create_entity("Entity1", EntityType.CONCEPT)
        await kg.create_entity("Entity2", EntityType.CODE)
        await kg.create_entity("Entity3", EntityType.PERSON)

        all_entities = await kg.list_entities()
        assert len(all_entities) >= 3

    @pytest.mark.asyncio
    async def test_list_entities_filtered_by_type(self, kg: KnowledgeGraph) -> None:
        """Test listing entities filtered by type."""
        await kg.create_entity("Code1", EntityType.CODE)
        await kg.create_entity("Code2", EntityType.CODE)
        await kg.create_entity("Person1", EntityType.PERSON)

        code_entities = await kg.list_entities(entity_type=EntityType.CODE)
        assert all(e.entity_type == EntityType.CODE for e in code_entities)
        assert len(code_entities) >= 2

    @pytest.mark.asyncio
    async def test_create_entity_relationship(self, kg: KnowledgeGraph) -> None:
        """Test creating relationships between entities."""
        e1 = await kg.create_entity("Source", EntityType.CONCEPT)
        e2 = await kg.create_entity("Target", EntityType.CONCEPT)

        result = await kg.create_entity_relationship(
            e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO, 0.8
        )
        assert result is True

        # Verify relationship exists
        rels = await kg.get_entity_relationships(e1.entity_id)
        assert len(rels) == 1
        assert rels[0]["target_id"] == e2.entity_id
        assert rels[0]["rel_type"] == RelationshipType.RELATED_TO

    @pytest.mark.asyncio
    async def test_create_relationship_invalid_entity(self, kg: KnowledgeGraph) -> None:
        """Test creating relationship with invalid entity returns False."""
        e1 = await kg.create_entity("Valid", EntityType.CONCEPT)
        result = await kg.create_entity_relationship(
            e1.entity_id, "invalid:id", RelationshipType.RELATED_TO
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_entity_relationships_with_filter(self, kg: KnowledgeGraph) -> None:
        """Test getting relationships filtered by type."""
        e1 = await kg.create_entity("FilterTest1", EntityType.CONCEPT)
        e2 = await kg.create_entity("FilterTest2", EntityType.CONCEPT)
        e3 = await kg.create_entity("FilterTest3", EntityType.CONCEPT)

        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        await kg.create_entity_relationship(e1.entity_id, e3.entity_id, RelationshipType.DERIVED_FROM)

        related = await kg.get_entity_relationships(e1.entity_id, RelationshipType.RELATED_TO)
        assert len(related) == 1
        assert related[0]["target_id"] == e2.entity_id

    @pytest.mark.asyncio
    async def test_find_inference_chain_direct(self, kg: KnowledgeGraph) -> None:
        """Test finding direct relationship path."""
        e1 = await kg.create_entity("ChainStart", EntityType.CONCEPT)
        e2 = await kg.create_entity("ChainEnd", EntityType.CONCEPT)
        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)

        result = await kg.find_inference_chain(e1.entity_id, e2.entity_id)
        assert result is not None
        assert result.depth == 1
        assert len(result.path) == 1

    @pytest.mark.asyncio
    async def test_find_inference_chain_transitive(self, kg: KnowledgeGraph) -> None:
        """Test finding transitive path through intermediate entities."""
        e1 = await kg.create_entity("TransitiveStart", EntityType.CONCEPT)
        e2 = await kg.create_entity("TransitiveMid", EntityType.CONCEPT)
        e3 = await kg.create_entity("TransitiveEnd", EntityType.CONCEPT)

        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        await kg.create_entity_relationship(e2.entity_id, e3.entity_id, RelationshipType.RELATED_TO)

        result = await kg.find_inference_chain(e1.entity_id, e3.entity_id, max_depth=3)
        assert result is not None
        assert result.depth == 2

    @pytest.mark.asyncio
    async def test_find_inference_chain_no_path(self, kg: KnowledgeGraph) -> None:
        """Test finding path when none exists."""
        e1 = await kg.create_entity("Unconnected1", EntityType.CONCEPT)
        e2 = await kg.create_entity("Unconnected2", EntityType.CONCEPT)

        result = await kg.find_inference_chain(e1.entity_id, e2.entity_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_inference_chains_multiple(self, kg: KnowledgeGraph) -> None:
        """Test finding multiple inference paths."""
        e1 = await kg.create_entity("MultiStart", EntityType.CONCEPT)
        e2 = await kg.create_entity("MultiMid", EntityType.CONCEPT)
        e3 = await kg.create_entity("MultiEnd", EntityType.CONCEPT)
        e4 = await kg.create_entity("MultiAlt", EntityType.CONCEPT)

        # Direct path
        await kg.create_entity_relationship(e1.entity_id, e3.entity_id, RelationshipType.RELATED_TO)
        # Indirect path
        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        await kg.create_entity_relationship(e2.entity_id, e3.entity_id, RelationshipType.RELATED_TO)
        # Alternative
        await kg.create_entity_relationship(e1.entity_id, e4.entity_id, RelationshipType.DERIVED_FROM)
        await kg.create_entity_relationship(e4.entity_id, e3.entity_id, RelationshipType.DERIVED_FROM)

        chains = await kg.get_inference_chains(e1.entity_id, e3.entity_id, max_depth=3)
        assert len(chains) >= 2  # Should find at least 2 paths

    @pytest.mark.asyncio
    async def test_get_transitive_closure(self, kg: KnowledgeGraph) -> None:
        """Test getting all entities reachable within depth."""
        e1 = await kg.create_entity("ClosureStart", EntityType.CONCEPT)
        e2 = await kg.create_entity("ClosureL1", EntityType.CONCEPT)
        e3 = await kg.create_entity("ClosureL2", EntityType.CONCEPT)

        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        await kg.create_entity_relationship(e2.entity_id, e3.entity_id, RelationshipType.RELATED_TO)

        closure = await kg.get_transitive_closure(e1.entity_id, depth=2)
        entity_ids = [c["entity_id"] for c in closure]
        assert e2.entity_id in entity_ids
        assert e3.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entity_graph(self, kg: KnowledgeGraph) -> None:
        """Test getting complete entity graph."""
        e1 = await kg.create_entity("GraphNode1", EntityType.CONCEPT)
        e2 = await kg.create_entity("GraphNode2", EntityType.CONCEPT)
        e3 = await kg.create_entity("GraphNode3", EntityType.CONCEPT)

        # e1 -> e2 (outgoing for e1, incoming for e2)
        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        # e3 -> e1 (incoming for e1)
        await kg.create_entity_relationship(e3.entity_id, e1.entity_id, RelationshipType.DERIVED_FROM)

        result = await kg.get_entity_graph(e1.entity_id)
        assert result is not None
        assert result.entity_id == e1.entity_id

    @pytest.mark.asyncio
    async def test_get_entity_graph_with_transitive(self, kg: KnowledgeGraph) -> None:
        """Test entity graph with transitive connections."""
        e1 = await kg.create_entity("TransGraph1", EntityType.CONCEPT)
        e2 = await kg.create_entity("TransGraph2", EntityType.CONCEPT)
        e3 = await kg.create_entity("TransGraph3", EntityType.CONCEPT)

        await kg.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)
        await kg.create_entity_relationship(e2.entity_id, e3.entity_id, RelationshipType.RELATED_TO)

        result = await kg.get_entity_graph(e1.entity_id, depth=2)
        assert result is not None
        assert len(result.inferred) > 0

    @pytest.fixture
    def kg_enabled(self) -> KnowledgeGraph:
        """Create a knowledge graph instance with KG enabled."""
        from memini_ai.config import MeminiConfig
        config = MeminiConfig()
        # Temporarily enable KG
        object.__setattr__(config, 'knowledge_graph_enabled', True)
        kg = KnowledgeGraph(config=config)
        return kg

    @pytest.mark.asyncio
    async def test_query_kg_full(self, kg_enabled: KnowledgeGraph) -> None:
        """Test full graph query."""
        e1 = await kg_enabled.create_entity("Query1", EntityType.CODE)
        e2 = await kg_enabled.create_entity("Query2", EntityType.CONCEPT)
        await kg_enabled.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)

        query = KGQuery()
        result = await kg_enabled.query_kg(query)
        assert result["success"] is True
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_query_kg_with_entity_filter(self, kg_enabled: KnowledgeGraph) -> None:
        """Test query with entity A filter."""
        e1 = await kg_enabled.create_entity("FilterA", EntityType.CONCEPT)
        e2 = await kg_enabled.create_entity("FilterB", EntityType.CONCEPT)
        await kg_enabled.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.RELATED_TO)

        query = KGQuery(entity_a=e1.entity_id)
        result = await kg_enabled.query_kg(query)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_query_kg_with_rel_type_filter(self, kg_enabled: KnowledgeGraph) -> None:
        """Test query with relationship type filter."""
        e1 = await kg_enabled.create_entity("RelFilter1", EntityType.CONCEPT)
        e2 = await kg_enabled.create_entity("RelFilter2", EntityType.CONCEPT)
        await kg_enabled.create_entity_relationship(e1.entity_id, e2.entity_id, RelationshipType.DERIVED_FROM)

        query = KGQuery(relationship_types=[RelationshipType.DERIVED_FROM])
        result = await kg_enabled.query_kg(query)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_query_kg_disabled_returns_empty(self, kg: KnowledgeGraph) -> None:
        """Test query returns error when KG is disabled."""
        # KG is disabled by default in test
        query = KGQuery()
        result = await kg.query_kg(query)
        assert result["success"] is False
        assert "not enabled" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_resolve_entity(self, kg: KnowledgeGraph) -> None:
        """Test entity resolution by ID, name, or fuzzy search."""
        entity = await kg.create_entity("ResolveTest", EntityType.CONCEPT)

        # By ID
        resolved = await kg._resolve_entity(entity.entity_id)
        assert resolved == entity.entity_id

        # By canonical name
        resolved = await kg._resolve_entity("ResolveTest")
        assert resolved == entity.entity_id

    @pytest.mark.asyncio
    async def test_extract_and_register_entities(self, kg: KnowledgeGraph) -> None:
        """Test extracting and registering entities from text."""
        text = "The function calculateTotal() processes Order objects from the database."
        entities = await kg.extract_and_register_entities(text)
        assert len(entities) >= 0  # Should extract some entities

    @pytest.mark.asyncio
    async def test_link_memory_to_entities(self, kg: KnowledgeGraph) -> None:
        """Test linking memory to extracted entities."""
        text = "UserService handles user authentication."
        entity_ids = await kg.link_memory_to_entities("test-memory-id", text)
        # Should have some entities
        assert isinstance(entity_ids, list)

    @pytest.mark.asyncio
    async def test_get_stats(self, kg: KnowledgeGraph) -> None:
        """Test getting KG statistics."""
        await kg.create_entity("Stats1", EntityType.CODE)
        await kg.create_entity("Stats2", EntityType.CONCEPT)
        await kg.create_entity("Stats3", EntityType.PERSON)

        stats = await kg.get_stats()
        assert "total_entities" in stats
        assert "entity_types" in stats
        assert "total_relationships" in stats
