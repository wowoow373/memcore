import logging
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.base import MemoryItem
from mem0.memory.search_engine import SearchEngine
from mem0.vector_stores.pgvector import OutputData


class TestImportBuildFilters:
    def test_import_build_filters(self):
        from mem0.memory.main import _build_filters_and_metadata

        assert callable(_build_filters_and_metadata)

    def test_import_normalize_timestamp(self):
        from mem0.memory.search_engine import _normalize_iso_timestamp_to_utc

        assert callable(_normalize_iso_timestamp_to_utc)

    def test_import_memory_item(self):
        from mem0.configs.base import MemoryItem

        assert hasattr(MemoryItem, "model_fields")

    def test_import_output_data(self):
        from mem0.vector_stores.pgvector import OutputData

        assert hasattr(OutputData, "model_fields")

    def test_import_process_telemetry_filters(self):
        from mem0.memory.utils import process_telemetry_filters

        assert callable(process_telemetry_filters)


class TestSearchEngineInit:
    def test_init_with_all_components(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()
        mock_graph = MagicMock()
        mock_reranker = MagicMock()

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            graph_store=mock_graph,
            reranker=mock_reranker,
        )

        assert engine.embedding_model is mock_embedder
        assert engine.vector_store is mock_vector
        assert engine.graph is mock_graph
        assert engine.reranker is mock_reranker
        assert engine.enable_graph is True
        assert engine.search_graph is not None

    def test_init_without_graph(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            graph_store=None,
            reranker=None,
        )

        assert engine.graph is None
        assert engine.enable_graph is False
        assert engine.reranker is None
        assert engine.search_graph is not None

    def test_init_without_reranker(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()
        mock_graph = MagicMock()

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            graph_store=mock_graph,
        )

        assert engine.reranker is None
        assert engine.enable_graph is True

    def test_init_minimal(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
        )

        assert engine.enable_graph is False
        assert engine.reranker is None
        assert engine.search_graph is not None


class TestEmbedQueryNode:
    def test_embed_query_normal(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        engine = SearchEngine(mock_embedder, MagicMock())

        result = engine._embed_query("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_embedder.embed.assert_called_once_with("hello world", "search")

    def test_embed_query_empty(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = []
        engine = SearchEngine(mock_embedder, MagicMock())

        result = engine._embed_query("")

        assert result == []

    def test_embed_query_propagates_error(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = RuntimeError("embed failed")
        engine = SearchEngine(mock_embedder, MagicMock())

        with pytest.raises(RuntimeError, match="embed failed"):
            engine._embed_query("test")


class TestVectorSearchNode:
    def test_vector_search_basic(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.95,
                payload={
                    "data": "Test memory content",
                    "hash": "abc123",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-02T00:00:00",
                    "user_id": "u1",
                },
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test",
            embeddings=[0.1, 0.2],
            filters={"user_id": "u1"},
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "mem-1"
        assert results[0]["memory"] == "Test memory content"
        assert results[0]["hash"] == "abc123"
        assert results[0]["score"] == 0.95
        assert results[0]["user_id"] == "u1"
        mock_vector.search.assert_called_once_with(
            query="test", vectors=[0.1, 0.2], limit=10, filters={"user_id": "u1"}
        )

    def test_vector_search_threshold_filtering_high(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.5,
                payload={"data": "Low score memory", "hash": "h1"},
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test",
            embeddings=[0.1],
            filters={},
            limit=10,
            threshold=0.9,
        )

        assert len(results) == 0

    def test_vector_search_threshold_filtering_low(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.5,
                payload={"data": "Medium score memory", "hash": "h1"},
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test",
            embeddings=[0.1],
            filters={},
            limit=10,
            threshold=0.3,
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.5

    def test_vector_search_empty_results(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test", embeddings=[0.1], filters={}, limit=10
        )

        assert results == []

    def test_vector_search_malformed_payload_missing_data(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.9,
                payload={"hash": "h1", "created_at": "2024-01-01T00:00:00"},
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test", embeddings=[0.1], filters={}, limit=10
        )

        assert len(results) == 1
        assert results[0]["memory"] == ""  # missing data falls back to empty

    def test_vector_search_skips_missing_payload(self, caplog):
        mock_vector = MagicMock()
        # Create a mock result without payload attribute
        bad_result = MagicMock()
        bad_result.id = "bad-mem"
        bad_result.score = 0.8
        del bad_result.payload  # no payload attribute

        good_result = OutputData(
            id="good-mem",
            score=0.9,
            payload={"data": "Good memory", "hash": "h2"},
        )
        mock_vector.search.return_value = [bad_result, good_result]
        engine = SearchEngine(MagicMock(), mock_vector)

        with caplog.at_level(logging.WARNING):
            results = engine._search_vector_store(
                query="test", embeddings=[0.1], filters={}, limit=10
            )

        assert len(results) == 1
        assert results[0]["id"] == "good-mem"
        assert "Skipping memory result with missing payload" in caplog.text

    def test_vector_search_timestamp_normalization(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.9,
                payload={
                    "data": "Test",
                    "created_at": "2024-01-01T12:00:00+05:30",
                    "updated_at": "2024-01-02T08:00:00+02:00",
                },
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test", embeddings=[0.1], filters={}, limit=10
        )

        # Timestamps should be normalized to UTC
        assert results[0]["created_at"].endswith("+00:00")
        assert results[0]["updated_at"].endswith("+00:00")

    def test_vector_search_additional_metadata(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.9,
                payload={
                    "data": "Test",
                    "hash": "h1",
                    "custom_key": "custom_value",
                    "another": 123,
                },
            )
        ]
        engine = SearchEngine(MagicMock(), mock_vector)

        results = engine._search_vector_store(
            query="test", embeddings=[0.1], filters={}, limit=10
        )

        assert results[0]["metadata"] == {"custom_key": "custom_value", "another": 123}


class TestGraphSearchNode:
    def test_graph_search_disabled_no_graph(self):
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=None)

        results = engine._search_graph("test", {"user_id": "u1"}, graph_depth=2)

        assert results == []

    def test_graph_search_depth_zero(self):
        mock_graph = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        results = engine._search_graph("test", {"user_id": "u1"}, graph_depth=0)

        assert results == []
        mock_graph.search.assert_not_called()

    def test_graph_search_with_depth(self):
        mock_graph = MagicMock()
        mock_graph.search.return_value = [
            {"source": "Alice", "relationship": "KNOWS", "destination": "Bob"}
        ]
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        results = engine._search_graph("test", {"user_id": "u1"}, graph_depth=2)

        assert len(results) == 1
        assert results[0]["source"] == "Alice"
        mock_graph.search.assert_called_once_with("test", {"user_id": "u1"}, limit=2)

    def test_graph_search_passes_depth_as_limit(self):
        mock_graph = MagicMock()
        mock_graph.search.return_value = []
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        engine._search_graph("test", {"user_id": "u1"}, graph_depth=5)

        mock_graph.search.assert_called_once_with("test", {"user_id": "u1"}, limit=5)

    def test_graph_search_propagates_error(self):
        mock_graph = MagicMock()
        mock_graph.search.side_effect = RuntimeError("graph error")
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        with pytest.raises(RuntimeError, match="graph error"):
            engine._search_graph("test", {"user_id": "u1"}, graph_depth=2)


class TestMergeResultsNode:
    def test_merge_empty_inputs(self):
        result = SearchEngine._merge_results([], [])

        assert result == {"vector_results": [], "graph_results": []}

    def test_merge_dedup_by_id_keeps_higher_score(self):
        vector_results = [
            {"id": "mem-1", "memory": "First", "score": 0.7},
            {"id": "mem-1", "memory": "Duplicate", "score": 0.9},
        ]
        graph_results = [
            {"source": "A", "relationship": "r", "destination": "B"}
        ]

        result = SearchEngine._merge_results(vector_results, graph_results)

        assert len(result["vector_results"]) == 1
        assert result["vector_results"][0]["score"] == 0.9
        assert result["graph_results"] == graph_results

    def test_merge_no_duplicates(self):
        vector_results = [
            {"id": "mem-1", "memory": "One", "score": 0.9},
            {"id": "mem-2", "memory": "Two", "score": 0.8},
            {"id": "mem-3", "memory": "Three", "score": 0.7},
        ]

        result = SearchEngine._merge_results(vector_results, [])

        assert len(result["vector_results"]) == 3

    def test_merge_preserves_graph_results(self):
        vector_results = [{"id": "mem-1", "memory": "V", "score": 0.9}]
        graph_results = [
            {"source": "A", "relationship": "r1", "destination": "B"},
            {"source": "C", "relationship": "r2", "destination": "D"},
        ]

        result = SearchEngine._merge_results(vector_results, graph_results)

        assert result["graph_results"] == graph_results

    def test_merge_missing_id_field(self, caplog):
        vector_results = [
            {"memory": "No id", "score": 0.9},
            {"id": "mem-1", "memory": "Has id", "score": 0.8},
        ]

        with caplog.at_level(logging.WARNING):
            result = SearchEngine._merge_results(vector_results, [])

        assert len(result["vector_results"]) == 1
        assert result["vector_results"][0]["id"] == "mem-1"
        assert "Skipping vector result with missing id" in caplog.text

    def test_merge_none_scores(self):
        vector_results = [
            {"id": "mem-1", "memory": "A", "score": None},
            {"id": "mem-1", "memory": "B", "score": 0.5},
        ]

        result = SearchEngine._merge_results(vector_results, [])

        # None score treated as 0.0, so 0.5 wins
        assert result["vector_results"][0]["score"] == 0.5


class TestRerankNode:
    def test_rerank_no_reranker(self):
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=None)
        vector_results = [{"id": "mem-1", "memory": "Test"}]

        result = engine._rerank_results("query", vector_results, limit=5)

        assert result == vector_results

    def test_rerank_empty_results(self):
        mock_reranker = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)

        result = engine._rerank_results("query", [], limit=5)

        assert result == []
        mock_reranker.rerank.assert_not_called()

    def test_rerank_success(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"memory": "Second", "id": "mem-2", "rerank_score": 0.95},
            {"memory": "First", "id": "mem-1", "rerank_score": 0.85},
        ]
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)

        vector_results = [
            {"id": "mem-1", "memory": "First"},
            {"id": "mem-2", "memory": "Second"},
        ]
        result = engine._rerank_results("query", vector_results, limit=2)

        assert len(result) == 2
        assert result[0]["id"] == "mem-2"
        assert result[0]["rerank_score"] == 0.95
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args[0][0] == "query"
        assert call_args[1]["top_k"] == 2

    def test_rerank_passes_correct_top_k(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)

        engine._rerank_results("query", [{"id": "m1", "memory": "test"}], limit=7)

        assert mock_reranker.rerank.call_args[1]["top_k"] == 7

    def test_rerank_propagates_error(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("rerank failed")
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)

        with pytest.raises(RuntimeError, match="rerank failed"):
            engine._rerank_results("query", [{"id": "m1", "memory": "test"}], limit=5)


class TestBuildResponseNode:
    def test_response_with_graph(self):
        mock_graph = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        merged = {
            "vector_results": [{"id": "mem-1", "memory": "Test"}],
            "graph_results": [{"source": "A", "relationship": "r", "destination": "B"}],
        }
        response = engine._build_search_response(merged)

        assert "results" in response
        assert "relations" in response
        assert response["results"] == merged["vector_results"]
        assert response["relations"] == merged["graph_results"]

    def test_response_without_graph(self):
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=None)

        merged = {
            "vector_results": [{"id": "mem-1", "memory": "Test"}],
            "graph_results": [],
        }
        response = engine._build_search_response(merged)

        assert "results" in response
        assert "relations" not in response
        assert response["results"] == merged["vector_results"]

    def test_response_empty_results(self):
        mock_graph = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        merged = {"vector_results": [], "graph_results": []}
        response = engine._build_search_response(merged)

        assert response["results"] == []
        assert response["relations"] == []


class TestConditionalEdges:
    def test_should_search_graph_true(self):
        mock_graph = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)
        state = {"graph_depth": 2}

        assert engine._should_search_graph(state) == "graph_search"

    def test_should_search_graph_depth_zero(self):
        mock_graph = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)
        state = {"graph_depth": 0}

        assert engine._should_search_graph(state) == "merge"

    def test_should_search_graph_disabled(self):
        engine = SearchEngine(MagicMock(), MagicMock(), graph_store=None)
        state = {"graph_depth": 2}

        assert engine._should_search_graph(state) == "merge"

    def test_should_rerank_true(self):
        mock_reranker = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)
        state = {
            "rerank": True,
            "merged_results": {"vector_results": [{"id": "m1"}]},
        }

        assert engine._should_rerank(state) == "rerank"

    def test_should_rerank_disabled_flag(self):
        mock_reranker = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)
        state = {
            "rerank": False,
            "merged_results": {"vector_results": [{"id": "m1"}]},
        }

        assert engine._should_rerank(state) == "build_response"

    def test_should_rerank_no_reranker(self):
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=None)
        state = {
            "rerank": True,
            "merged_results": {"vector_results": [{"id": "m1"}]},
        }

        assert engine._should_rerank(state) == "build_response"

    def test_should_rerank_empty_results(self):
        mock_reranker = MagicMock()
        engine = SearchEngine(MagicMock(), MagicMock(), reranker=mock_reranker)
        state = {
            "rerank": True,
            "merged_results": {"vector_results": []},
        }

        assert engine._should_rerank(state) == "build_response"


class TestLangGraphCompilation:
    def test_search_graph_compilation(self):
        engine = SearchEngine(MagicMock(), MagicMock())

        # Should compile without errors
        assert engine.search_graph is not None

    def test_langgraph_structure(self):
        engine = SearchEngine(MagicMock(), MagicMock())
        graph = engine.search_graph.get_graph()
        nodes = list(graph.nodes.keys())

        expected_nodes = [
            "embed",
            "vector_search",
            "graph_search",
            "merge",
            "rerank",
            "build_response",
        ]
        for node in expected_nodes:
            assert node in nodes, f"Node {node} not found in compiled graph"


class TestSearchEntryPoint:
    def test_search_full_pipeline_mocked(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.95,
                payload={"data": "Test memory", "hash": "h1"},
            )
        ]
        mock_graph = MagicMock()
        mock_graph.search.return_value = [
            {"source": "A", "relationship": "r", "destination": "B"}
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"memory": "Test memory", "id": "mem-1", "rerank_score": 0.99}
        ]

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            graph_store=mock_graph,
            reranker=mock_reranker,
        )

        result = engine.search(
            query="test query",
            filters={"user_id": "u1"},
            limit=10,
            graph_depth=2,
            rerank=True,
        )

        assert "results" in result
        assert "relations" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "mem-1"
        assert len(result["relations"]) == 1

    def test_search_vector_only_no_graph(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1]
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.9,
                payload={"data": "Only vector", "hash": "h1"},
            )
        ]

        engine = SearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
        )

        result = engine.search(query="test", filters={"user_id": "u1"})

        assert "results" in result
        assert "relations" not in result
        assert len(result["results"]) == 1

    def test_search_no_rerank(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1]
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            OutputData(
                id="mem-1",
                score=0.9,
                payload={"data": "No rerank", "hash": "h1"},
            )
        ]

        engine = SearchEngine(MagicMock(), mock_vector)
        result = engine.search(query="test", filters={}, rerank=False)

        assert len(result["results"]) == 1
        assert "rerank_score" not in result["results"][0]


class TestMemorySearchIntegration:
    """Integration tests for Memory.search() delegating to SearchEngine."""

    def _create_memory_instance(self):
        """Helper to create a Memory instance with mocked dependencies."""
        from unittest.mock import Mock, patch

        with (
            patch("mem0.memory.main.EmbedderFactory") as mock_embedder,
            patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
            patch("mem0.memory.main.LlmFactory") as mock_llm,
            patch("mem0.memory.telemetry.capture_event"),
            patch("mem0.memory.graph_memory.MemoryGraph"),
            patch("mem0.memory.main.GraphStoreFactory") as mock_graph_store,
        ):
            mock_embedder.create.return_value = Mock()
            mock_vector_store.create.return_value = Mock()
            mock_vector_store.create.return_value.search.return_value = []
            mock_llm.create.return_value = Mock()

            mock_graph_instance = Mock()
            mock_graph_store.create.return_value = mock_graph_instance

            from mem0.configs.base import MemoryConfig
            from mem0.memory.main import Memory

            config = MemoryConfig(version="v1.1")
            config.graph_store.config = {"some_config": "value"}
            return Memory(config)

    def test_memory_search_delegates_to_engine(self):
        memory = self._create_memory_instance()
        memory.search_engine.search = MagicMock(return_value={
            "results": [{"id": "mem-1", "memory": "test"}],
            "relations": [{"source": "A", "relationship": "r", "destination": "B"}],
        })

        result = memory.search("test query", user_id="u1")

        memory.search_engine.search.assert_called_once()
        call_kwargs = memory.search_engine.search.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["filters"]["user_id"] == "u1"
        assert call_kwargs["limit"] == 100
        assert call_kwargs["threshold"] is None
        assert call_kwargs["graph_depth"] == 2
        assert call_kwargs["rerank"] is True
        assert result["results"][0]["id"] == "mem-1"

    def test_memory_search_e2e_vector_only(self):
        memory = self._create_memory_instance()
        # Disable graph for this test
        memory.search_engine.enable_graph = False
        memory.search_engine.graph = None

        memory.embedding_model.embed = MagicMock(return_value=[0.1, 0.2])
        memory.vector_store.search = MagicMock(return_value=[
            OutputData(
                id="mem-1",
                score=0.9,
                payload={"data": "Vector only memory", "hash": "h1", "user_id": "u1"},
            )
        ])

        result = memory.search("test", user_id="u1")

        assert "results" in result
        assert "relations" not in result
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "Vector only memory"

    def test_memory_search_e2e_with_graph(self):
        memory = self._create_memory_instance()

        memory.embedding_model.embed = MagicMock(return_value=[0.1, 0.2])
        memory.vector_store.search = MagicMock(return_value=[
            OutputData(
                id="mem-1",
                score=0.9,
                payload={"data": "Memory with graph", "hash": "h1", "user_id": "u1"},
            )
        ])
        memory.graph.search = MagicMock(return_value=[
            {"source": "Alice", "relationship": "KNOWS", "destination": "Bob"}
        ])

        result = memory.search("test", user_id="u1")

        assert "results" in result
        assert "relations" in result
        assert len(result["results"]) == 1
        assert len(result["relations"]) == 1
        assert result["relations"][0]["source"] == "Alice"

    def test_memory_search_e2e_with_rerank(self):
        memory = self._create_memory_instance()
        memory.search_engine.enable_graph = False
        memory.search_engine.graph = None

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"memory": "Second", "id": "mem-2", "rerank_score": 0.95},
            {"memory": "First", "id": "mem-1", "rerank_score": 0.85},
        ]
        memory.search_engine.reranker = mock_reranker

        memory.embedding_model.embed = MagicMock(return_value=[0.1])
        memory.vector_store.search = MagicMock(return_value=[
            OutputData(id="mem-1", score=0.8, payload={"data": "First", "hash": "h1"}),
            OutputData(id="mem-2", score=0.7, payload={"data": "Second", "hash": "h2"}),
        ])

        result = memory.search("test", user_id="u1", rerank=True)

        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "mem-2"
        assert result["results"][0]["rerank_score"] == 0.95
        mock_reranker.rerank.assert_called_once()
