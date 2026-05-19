from unittest.mock import MagicMock

import pytest

from mem0.memory.process_search_engine import ProcessMemorySearchEngine


def _make_output_data(id, score, payload):
    """Create a mock vector store result object."""
    result = MagicMock()
    result.id = id
    result.score = score
    result.payload = payload
    return result


# ══════════════════════════════════════════════════════════════════════
# Constructor tests
# ══════════════════════════════════════════════════════════════════════


class TestProcessMemorySearchEngineInit:
    def test_init_with_all_deps(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()
        mock_graph = MagicMock()

        engine = ProcessMemorySearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            graph_store=mock_graph,
        )

        assert engine.embedding_model is mock_embedder
        assert engine.vector_store is mock_vector
        assert engine.graph_store is mock_graph

    def test_init_without_graph_store(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()

        engine = ProcessMemorySearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
        )

        assert engine.graph_store is None

    def test_init_minimal(self):
        mock_embedder = MagicMock()
        mock_vector = MagicMock()

        engine = ProcessMemorySearchEngine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
        )

        assert engine.embedding_model is mock_embedder
        assert engine.vector_store is mock_vector


# ══════════════════════════════════════════════════════════════════════
# _embed tests
# ══════════════════════════════════════════════════════════════════════


class TestEmbed:
    def test_embed_valid_text(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock())

        result = engine._embed("hello world")

        mock_embedder.embed.assert_called_once_with("hello world", "search")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_empty_string(self):
        mock_embedder = MagicMock()
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock())

        result = engine._embed("")

        mock_embedder.embed.assert_not_called()
        assert result == []

    def test_embed_whitespace_string(self):
        mock_embedder = MagicMock()
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock())

        result = engine._embed("   ")

        mock_embedder.embed.assert_not_called()
        assert result == []


# ══════════════════════════════════════════════════════════════════════
# _search_graph_by_brief tests
# ══════════════════════════════════════════════════════════════════════


class TestSearchGraphByBrief:
    def test_brief_match_returns_nodes(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_graph = MagicMock()
        mock_graph.search_nodes_by_embedding.return_value = [
            {"name": "01 - Read main.py", "brief": "...", "goal": "auth", "step": "01 - Read main.py", "score": 0.92}
        ]
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock(), graph_store=mock_graph)

        result = engine._search_graph_by_brief("Read main.py", {"user_id": "u1"})

        mock_graph.search_nodes_by_embedding.assert_called_once_with(
            [0.1] * 10, {"user_id": "u1"}, top_k=10
        )
        assert len(result) == 1
        assert result[0]["name"] == "01 - Read main.py"

    def test_empty_brief_returns_empty(self):
        mock_graph = MagicMock()
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        result = engine._search_graph_by_brief("", {"user_id": "u1"})

        assert result == []
        mock_graph.search_nodes_by_embedding.assert_not_called()

    def test_graph_store_none_returns_empty(self):
        mock_embedder = MagicMock()
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock(), graph_store=None)

        result = engine._search_graph_by_brief("some brief", {"user_id": "u1"})

        assert result == []

    def test_no_matching_nodes(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 10
        mock_graph = MagicMock()
        mock_graph.search_nodes_by_embedding.return_value = []
        engine = ProcessMemorySearchEngine(mock_embedder, MagicMock(), graph_store=mock_graph)

        result = engine._search_graph_by_brief("unknown", {"user_id": "u1"})

        assert result == []


# ══════════════════════════════════════════════════════════════════════
# _expand_neighbors tests
# ══════════════════════════════════════════════════════════════════════


class TestExpandNeighbors:
    def test_expands_from_node_names(self):
        mock_graph = MagicMock()
        mock_graph.search_nodes.return_value = [
            {"source": "01 - Read main.py", "relationship": "DEPENDS_ON", "destination": "03 - Create auth.py"}
        ]
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        result = engine._expand_neighbors(
            ["01 - Read main.py"], {"user_id": "u1"}, depth=1
        )

        mock_graph.search_nodes.assert_called_once_with(
            ["01 - Read main.py"], {"user_id": "u1"}, depth=1
        )
        assert len(result) == 1
        assert result[0]["source"] == "01 - Read main.py"

    def test_empty_node_names(self):
        mock_graph = MagicMock()
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        result = engine._expand_neighbors([], {"user_id": "u1"})

        assert result == []
        mock_graph.search_nodes.assert_not_called()

    def test_graph_store_none(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=None)

        result = engine._expand_neighbors(["node1"], {"user_id": "u1"})

        assert result == []


# ══════════════════════════════════════════════════════════════════════
# _search_graph_by_step_names tests
# ══════════════════════════════════════════════════════════════════════


class TestSearchGraphByStepNames:
    def test_traverses_full_chain(self):
        mock_graph = MagicMock()
        mock_graph.search_nodes.return_value = [
            {"source": "01 - Read", "relationship": "DEPENDS_ON", "destination": "03 - Auth"},
            {"source": "03 - Auth", "relationship": "DEPENDS_ON", "destination": "05 - Test"},
        ]
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        result = engine._search_graph_by_step_names(
            ["03 - Auth", "05 - Test"], {"user_id": "u1"}
        )

        mock_graph.search_nodes.assert_called_once_with(
            ["03 - Auth", "05 - Test"], {"user_id": "u1"}, depth=10
        )
        assert len(result) == 2

    def test_empty_step_names(self):
        mock_graph = MagicMock()
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=mock_graph)

        result = engine._search_graph_by_step_names([], {"user_id": "u1"})

        assert result == []

    def test_graph_store_none(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock(), graph_store=None)

        result = engine._search_graph_by_step_names(["node1"], {"user_id": "u1"})

        assert result == []


# ══════════════════════════════════════════════════════════════════════
# _semantic_filter tests
# ══════════════════════════════════════════════════════════════════════


class TestSemanticFilter:
    def _make_fixed_embedder(self):
        """Return an embedder that maps known strings to fixed vectors."""
        embedder = MagicMock()

        def embed_side_effect(text, action):
            if action != "search":
                return []
            if "reading" in text.lower():
                return [1.0, 0.0, 0.0]
            if "writing" in text.lower():
                return [0.0, 1.0, 0.0]
            if "testing" in text.lower():
                return [0.0, 0.0, 1.0]
            if "debugging" in text.lower():
                return [0.5, 0.5, 0.0]
            return [0.0, 0.0, 0.0]

        embedder.embed.side_effect = embed_side_effect
        return embedder

    def test_filters_by_threshold(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())

        nodes = [
            {"name": "01", "brief": "reading main.py"},
            {"name": "02", "brief": "writing auth.py"},
            {"name": "03", "brief": "testing api"},
        ]
        previous_step = {"Brief": "reading config"}

        result = engine._semantic_filter(nodes, previous_step, threshold=0.5)

        # "reading main.py" matches "reading config" (cosine=1.0)
        # "writing auth.py" has cosine 0.0 with "reading config"
        # "testing api" has cosine 0.0
        # "debugging" has cosine ~0.707 with "reading config" — but not in nodes
        assert len(result) == 1
        assert result[0]["name"] == "01"

    def test_previous_step_none_returns_unchanged(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        nodes = [{"name": "01", "brief": "reading main.py"}]

        result = engine._semantic_filter(nodes, None)

        assert result == nodes

    def test_previous_step_no_brief_key_returns_unchanged(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        nodes = [{"name": "01", "brief": "reading main.py"}]
        previous_step = {"Goal": "do stuff", "Step": "01"}

        result = engine._semantic_filter(nodes, previous_step)

        assert result == nodes

    def test_empty_nodes(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        previous_step = {"Brief": "reading config"}

        result = engine._semantic_filter([], previous_step)

        assert result == []

    def test_all_below_threshold(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        nodes = [{"name": "02", "brief": "writing auth.py"}]
        previous_step = {"Brief": "reading config"}

        result = engine._semantic_filter(nodes, previous_step, threshold=0.9)

        assert result == []

    def test_node_without_brief_key_excluded(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        nodes = [
            {"name": "01", "brief": "reading main.py"},
            {"name": "02"},  # no brief
        ]
        previous_step = {"Brief": "reading config"}

        result = engine._semantic_filter(nodes, previous_step, threshold=0.5)

        assert len(result) == 1
        assert result[0]["name"] == "01"

    def test_sorted_by_similarity_desc(self):
        embedder = self._make_fixed_embedder()
        engine = ProcessMemorySearchEngine(embedder, MagicMock())
        nodes = [
            {"name": "02", "brief": "writing auth.py"},
            {"name": "04", "brief": "debugging middleware"},
        ]
        # previous is "reading config" [1,0,0]
        # "debugging middleware" [0.5,0.5,0] cosine with [1,0,0] ~= 0.707
        # "writing auth.py" [0,1,0] cosine with [1,0,0] = 0.0
        previous_step = {"Brief": "reading config"}

        result = engine._semantic_filter(nodes, previous_step, threshold=0.0)

        assert len(result) == 2
        assert result[0]["name"] == "04"  # higher similarity first
        assert result[1]["name"] == "02"


# ══════════════════════════════════════════════════════════════════════
# _cosine_similarity tests
# ══════════════════════════════════════════════════════════════════════


class TestCosineSimilarity:
    def test_identical_vectors(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 1.0

    def test_orthogonal_vectors(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert result == 0.0

    def test_zero_vector(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 0.0

    def test_empty_input(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([], [1.0, 0.0])
        assert result == 0.0

    def test_mismatched_lengths(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 0.0

    def test_opposite_vectors(self):
        engine = ProcessMemorySearchEngine(MagicMock(), MagicMock())
        result = engine._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == -1.0


# ══════════════════════════════════════════════════════════════════════
# _search_chunks tests
# ══════════════════════════════════════════════════════════════════════


class TestSearchChunks:
    def test_searches_with_memory_type_filter(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)

        engine._search_chunks("Add auth", {"user_id": "u1"})

        call_kwargs = mock_vector.search.call_args.kwargs
        assert call_kwargs["filters"]["memory_type"] == "process_chunk"
        assert call_kwargs["filters"]["user_id"] == "u1"

    def test_formats_results_correctly(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            _make_output_data(
                id="chunk_1",
                score=0.85,
                payload={
                    "goal": "Add user auth",
                    "steps": [{"step": "03 - Create auth.py", "brief": "..."}],
                    "memory_type": "process_chunk",
                    "data": "Add user auth",
                    "hash": "abc123",
                    "user_id": "u1",
                },
            )
        ]
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)

        result = engine._search_chunks("Add user auth", {"user_id": "u1"})

        assert len(result) == 1
        assert result[0]["goal"] == "Add user auth"
        assert result[0]["score"] == 0.85
        assert len(result[0]["steps"]) == 1
        assert result[0]["id"] == "chunk_1"
        assert "memory_type" not in result[0]["metadata"]
        assert "data" not in result[0]["metadata"]
        assert "hash" not in result[0]["metadata"]

    def test_empty_goal_returns_empty(self):
        mock_vector = MagicMock()
        engine = ProcessMemorySearchEngine(MagicMock(), mock_vector)

        result = engine._search_chunks("", {"user_id": "u1"})

        assert result == []
        mock_vector.search.assert_not_called()

    def test_does_not_mutate_input_filters(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)
        original_filters = {"user_id": "u1"}

        engine._search_chunks("goal", original_filters)

        assert "memory_type" not in original_filters

    def test_handles_search_error(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.side_effect = RuntimeError("db down")
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)

        result = engine._search_chunks("goal", {"user_id": "u1"})

        assert result == []


# ══════════════════════════════════════════════════════════════════════
# _search_summaries tests
# ══════════════════════════════════════════════════════════════════════


class TestSearchSummaries:
    def test_searches_with_memory_type_filter(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)

        engine._search_summaries("task desc", {"user_id": "u1"})

        call_kwargs = mock_vector.search.call_args.kwargs
        assert call_kwargs["filters"]["memory_type"] == "process_summary"
        assert call_kwargs["filters"]["user_id"] == "u1"

    def test_formats_results_correctly(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 5
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            _make_output_data(
                id="sum_1",
                score=0.78,
                payload={
                    "task_description": "Implement auth system",
                    "full_chain": [{"step": "01", "brief": "..."}, {"step": "02", "brief": "..."}],
                    "memory_type": "process_summary",
                    "data": "Implement auth system",
                    "hash": "def456",
                    "user_id": "u1",
                },
            )
        ]
        engine = ProcessMemorySearchEngine(mock_embedder, mock_vector)

        result = engine._search_summaries("Implement auth system", {"user_id": "u1"})

        assert len(result) == 1
        assert result[0]["task_description"] == "Implement auth system"
        assert result[0]["score"] == 0.78
        assert len(result[0]["full_chain"]) == 2
        assert result[0]["id"] == "sum_1"
        assert "memory_type" not in result[0]["metadata"]
        assert "data" not in result[0]["metadata"]

    def test_empty_task_desc_returns_empty(self):
        mock_vector = MagicMock()
        engine = ProcessMemorySearchEngine(MagicMock(), mock_vector)

        result = engine._search_summaries("", {"user_id": "u1"})

        assert result == []
        mock_vector.search.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# search_for_step tests (Flow 2)
# ══════════════════════════════════════════════════════════════════════


class TestSearchForStep:
    def _make_engine(self, with_graph=True):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 5
        vector = MagicMock()
        vector.search.return_value = []
        graph = MagicMock() if with_graph else None
        if graph:
            graph.search_nodes_by_embedding.return_value = []
            graph.search_nodes.return_value = []
        return embedder, vector, graph, ProcessMemorySearchEngine(embedder, vector, graph_store=graph)

    def test_full_flow_with_graph(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=True)
        graph.search_nodes_by_embedding.return_value = [
            {"name": "03 - Create auth.py", "brief": "create auth file", "goal": "Add auth", "step": "03 - Create auth.py", "score": 0.9}
        ]
        graph.search_nodes.return_value = [
            {"source": "01", "relationship": "DEPENDS_ON", "destination": "03"}
        ]
        def vector_search_side_effect(query=None, vectors=None, limit=None, filters=None):
            if filters and filters.get("memory_type") == "process_chunk":
                return [_make_output_data("c1", 0.8, {"goal": "Add auth", "steps": [], "memory_type": "process_chunk"})]
            elif filters and filters.get("memory_type") == "process_summary":
                return [_make_output_data("s1", 0.7, {"task_description": "full task", "full_chain": [], "memory_type": "process_summary"})]
            return []

        vector.search.side_effect = vector_search_side_effect

        current_step = {
            "Goal": "Add auth",
            "Step": "03 - Create auth.py",
            "Action": "create_file()",
            "Brief": "create auth file",
        }

        result = engine.search_for_step(current_step, filters={"user_id": "u1"})

        assert len(result["graph"]["matched_nodes"]) == 1
        assert len(result["graph"]["expanded_nodes"]) == 1
        assert len(result["chunks"]) == 1
        assert len(result["summaries"]) == 1

    def test_full_flow_without_graph(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=False)
        vector.search.return_value = [
            _make_output_data("c1", 0.8, {"goal": "Add auth", "steps": [], "memory_type": "process_chunk"}),
        ]

        current_step = {"Goal": "Add auth", "Brief": "create auth file"}
        result = engine.search_for_step(current_step, filters={"user_id": "u1"})

        assert result["graph"] == {"matched_nodes": [], "expanded_nodes": [], "filtered_nodes": []}
        assert len(result["chunks"]) == 1

    def test_task_estimate_fallback(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=False)
        vector.search.return_value = []

        current_step = {"Goal": "Add auth", "Brief": "create auth file"}
        engine._search_summaries = MagicMock(return_value=[])

        engine.search_for_step(current_step, filters={"user_id": "u1"})

        engine._search_summaries.assert_called_once()
        call_text = engine._search_summaries.call_args[0][0]
        assert "Add auth" in call_text
        assert "create auth file" in call_text

    def test_previous_step_triggers_semantic_filter(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=True)
        graph.search_nodes_by_embedding.return_value = [
            {"name": "03", "brief": "reading main.py", "goal": "Add auth", "step": "03", "score": 0.9},
            {"name": "04", "brief": "writing auth.py", "goal": "Add auth", "step": "04", "score": 0.8},
        ]
        graph.search_nodes.return_value = []

        # Embedder: reading->[1,0], writing->[0,1], testing->[0,0]
        embedder.embed.side_effect = lambda text, action: {
            "reading main.py": [1.0, 0.0],
            "writing auth.py": [0.0, 1.0],
            "reading config": [1.0, 0.0],
            "create auth file": [0.5, 0.5],
            "Add auth": [0.5, 0.5],
        }.get(text, [0.0, 0.0])

        current_step = {"Goal": "Add auth", "Brief": "create auth file", "Step": "05"}
        previous_step = {"Brief": "reading config"}

        result = engine.search_for_step(
            current_step, previous_step=previous_step, filters={"user_id": "u1"}
        )

        assert len(result["graph"]["filtered_nodes"]) == 1
        assert result["graph"]["filtered_nodes"][0]["name"] == "03"

    def test_empty_current_step(self):
        _, _, _, engine = self._make_engine(with_graph=False)

        result = engine.search_for_step({})

        assert result["graph"] == {"matched_nodes": [], "expanded_nodes": [], "filtered_nodes": []}
        assert result["chunks"] == []
        assert result["summaries"] == []


# ══════════════════════════════════════════════════════════════════════
# search_for_dedup tests (Flow 1)
# ══════════════════════════════════════════════════════════════════════


class TestSearchForDedup:
    def _make_engine(self, with_graph=True):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 5
        vector = MagicMock()
        vector.search.return_value = []
        graph = MagicMock() if with_graph else None
        if graph:
            graph.search_nodes.return_value = []
        return embedder, vector, graph, ProcessMemorySearchEngine(embedder, vector, graph_store=graph)

    def test_chunk_then_graph_dependency(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=True)
        vector.search.return_value = [
            _make_output_data(
                "c1", 0.9,
                {
                    "goal": "Add auth",
                    "steps": [
                        {"step": "03 - Create auth.py", "brief": "..."},
                        {"step": "04 - Modify main.py", "brief": "..."},
                    ],
                    "memory_type": "process_chunk",
                },
            )
        ]
        graph.search_nodes.return_value = [
            {"source": "01", "relationship": "DEPENDS_ON", "destination": "03"}
        ]

        result = engine.search_for_dedup(
            goals=["Add auth"], task_description="build auth system", filters={"user_id": "u1"}
        )

        # Graph was called with step names extracted from chunk
        graph.search_nodes.assert_called_once()
        call_args = graph.search_nodes.call_args[0]
        step_names = call_args[0]
        assert "03 - Create auth.py" in step_names
        assert "04 - Modify main.py" in step_names
        assert "chains" in result["graph"]

    def test_summary_independent(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=False)
        vector.search.return_value = []

        # Spy on _search_summaries
        engine._search_summaries = MagicMock(return_value=[])

        engine.search_for_dedup(
            goals=[], task_description="build auth system", filters={"user_id": "u1"}
        )

        engine._search_summaries.assert_called_once_with(
            "build auth system", {"user_id": "u1"}, top_k=3
        )

    def test_deduplicates_chunks_by_id(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=False)
        shared_payload = {
            "goal": "Add auth",
            "steps": [{"step": "03 - Create auth.py", "brief": "..."}],
            "memory_type": "process_chunk",
        }
        vector.search.return_value = [
            _make_output_data("c1", 0.9, shared_payload),
        ]
        # Both goals produce the same chunk
        result = engine.search_for_dedup(
            goals=["Add auth", "Add authentication"], filters={"user_id": "u1"}
        )

        # Should be deduplicated to 1
        assert len(result["chunks"]) == 1

    def test_task_description_fallback(self):
        embedder, vector, graph, engine = self._make_engine(with_graph=False)
        vector.search.return_value = []

        engine._search_summaries = MagicMock(return_value=[])

        engine.search_for_dedup(
            goals=["Add auth", "Setup DB"], filters={"user_id": "u1"}
        )

        call_text = engine._search_summaries.call_args[0][0]
        assert "Add auth" in call_text
        assert "Setup DB" in call_text

    def test_empty_goals(self):
        _, _, _, engine = self._make_engine(with_graph=False)

        result = engine.search_for_dedup(goals=[])

        assert result["chunks"] == []
        assert result["summaries"] == []
        assert result["graph"]["chains"] == []

    def test_graph_store_none_skips_graph(self):
        _, vector, _, engine = self._make_engine(with_graph=False)
        vector.search.return_value = [
            _make_output_data(
                "c1", 0.9,
                {"goal": "Add auth", "steps": [{"step": "03 - Create auth.py", "brief": "..."}], "memory_type": "process_chunk"},
            )
        ]

        result = engine.search_for_dedup(goals=["Add auth"], filters={"user_id": "u1"})

        assert len(result["chunks"]) == 1
        assert result["graph"]["chains"] == []
