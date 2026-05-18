import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from mem0.configs.base import MemoryItem
from mem0.graphs.tools import EXTRACT_ENTITIES_STRUCT_TOOL, EXTRACT_ENTITIES_TOOL

logger = logging.getLogger(__name__)


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


class SearchState(TypedDict):
    """LangGraph state schema for SearchEngine."""

    query: str
    filters: dict
    limit: int
    threshold: Optional[float]
    graph_depth: int
    rerank: bool
    embedding: Optional[List[float]]
    vector_results: List[dict]
    graph_results: List[dict]
    merged_results: dict
    final_results: dict
    error: Optional[str]


class SearchEngine:
    """Search unified recall engine.

    Orchestrates vector search, graph traversal, merge/dedup, and optional reranking.
    Internal LangGraph state machine; exposed via simple ``search()`` method.
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        graph_store: Optional[Any] = None,
        reranker: Optional[Any] = None,
        llm: Optional[Any] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.graph = graph_store
        self.reranker = reranker
        self.llm = llm
        self.enable_graph = graph_store is not None

        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(SearchState)
        builder.add_node("embed", self._node_embed)
        builder.add_node("vector_search", self._node_vector_search)
        builder.add_node("graph_search", self._node_graph_search)
        builder.add_node("merge", self._node_merge)
        builder.add_node("rerank", self._node_rerank)
        builder.add_node("build_response", self._node_build_response)

        builder.add_edge(START, "embed")
        builder.add_edge("embed", "vector_search")
        builder.add_conditional_edges(
            "vector_search",
            self._should_search_graph,
            {"graph_search": "graph_search", "merge": "merge"},
        )
        builder.add_edge("graph_search", "merge")
        builder.add_conditional_edges(
            "merge",
            self._should_rerank,
            {"rerank": "rerank", "build_response": "build_response"},
        )
        builder.add_edge("rerank", "build_response")
        builder.add_edge("build_response", END)

        self.search_graph = builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict,
        limit: int = 100,
        threshold: Optional[float] = None,
        graph_depth: int = 2,
        rerank: bool = True,
    ) -> dict:
        state: SearchState = {
            "query": query,
            "filters": filters,
            "limit": limit,
            "threshold": threshold,
            "graph_depth": graph_depth,
            "rerank": rerank,
            "embedding": None,
            "vector_results": [],
            "graph_results": [],
            "merged_results": {},
            "final_results": {},
            "error": None,
        }
        result = self.search_graph.invoke(state)
        return result["final_results"]

    # ------------------------------------------------------------------
    # Core node methods (testable independently, no LangGraph dependency)
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> List[float]:
        """Convert query text to embedding vector."""
        return self.embedding_model.embed(query, "search")

    def _search_vector_store(
        self,
        query: str,
        embeddings: List[float],
        filters: dict,
        limit: int,
        threshold: Optional[float] = None,
    ) -> List[dict]:
        """Search vector store and format results into MemoryItem dicts."""
        memories = self.vector_store.search(
            query=query, vectors=embeddings, limit=limit, filters=filters
        )
        return self._format_vector_results(memories, threshold)

    @staticmethod
    def _format_vector_results(
        memories: List[Any], threshold: Optional[float] = None
    ) -> List[dict]:
        """Format raw vector-store results into MemoryItem dicts with threshold filtering.

        Args:
            memories: List of result objects with ``id``, ``score``, ``payload`` attrs.
            threshold: Minimum score to include; ``None`` means include all.

        Returns:
            List of MemoryItem dicts.
        """
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "id",
            *promoted_payload_keys,
        }

        original_memories = []
        for mem in memories:
            if not hasattr(mem, "payload") or mem.payload is None:
                logger.warning("Skipping memory result with missing payload: %s", getattr(mem, "id", None))
                continue

            payload = mem.payload

            memory_item_dict = MemoryItem(
                id=getattr(mem, "id", None) or payload.get("id", ""),
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=_normalize_iso_timestamp_to_utc(payload.get("created_at")),
                updated_at=_normalize_iso_timestamp_to_utc(payload.get("updated_at")),
                score=getattr(mem, "score", None),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {
                k: v for k, v in payload.items() if k not in core_and_promoted_keys
            }
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            mem_score = getattr(mem, "score", None)
            if threshold is None or (mem_score is not None and mem_score >= threshold):
                original_memories.append(memory_item_dict)

        return original_memories

    def _extract_nodes(self, query: str, filters: dict) -> List[str]:
        """Extract entity node names from query text using LLM tool-call.

        Mirrors the logic previously in MemoryGraph._retrieve_nodes_from_data().
        """
        if self.llm is None:
            return []

        # Determine whether to use structured tool format based on LLM class name.
        llm_cls_name = type(self.llm).__name__.lower()
        if "structured" in llm_cls_name:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        else:
            _tools = [EXTRACT_ENTITIES_TOOL]

        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a smart assistant who understands entities and their types in a given text. "
                        f"If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters.get('user_id', 'user')} as the source entity. "
                        "Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question."
                    ),
                },
                {"role": "user", "content": query},
            ],
            tools=_tools,
        )

        entity_type_map = {}
        try:
            for tool_call in search_results.get("tool_calls", []):
                if tool_call.get("name") != "extract_entities":
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    entity_type_map[item["entity"]] = item.get("entity_type", "")
        except Exception as e:
            logger.warning(f"Error extracting nodes from query: {e}")

        # Normalize: lower + space → underscore
        nodes = [k.lower().replace(" ", "_") for k in entity_type_map.keys()]
        return nodes

    def _search_graph(
        self, query: str, filters: dict, graph_depth: int
    ) -> List[dict]:
        """Search graph store for related entities up to ``graph_depth`` hops."""
        if graph_depth <= 0 or not self.enable_graph:
            return []

        # Prefer new search_nodes interface (Neo4j)
        if hasattr(self.graph, "search_nodes"):
            if self.llm is not None:
                node_names = self._extract_nodes(query, filters)
            else:
                # Fallback: simple split when no LLM available
                if "," in query:
                    node_names = [n.strip() for n in query.split(",") if n.strip()]
                else:
                    node_names = [query.strip()]
            return self.graph.search_nodes(node_names, filters, depth=graph_depth)

        # Legacy fallback for other graph stores
        return self.graph.search(query, filters, limit=graph_depth)

    @staticmethod
    def _merge_results(
        vector_results: List[dict], graph_results: List[dict]
    ) -> dict:
        """Merge vector and graph results, deduplicating vector results by ID."""
        deduped = {}
        for item in vector_results:
            item_id = item.get("id")
            if item_id is None:
                logger.warning("Skipping vector result with missing id during merge")
                continue
            existing = deduped.get(item_id)
            if existing is None:
                deduped[item_id] = item
            else:
                existing_score = existing.get("score") or 0.0
                item_score = item.get("score") or 0.0
                if item_score > existing_score:
                    deduped[item_id] = item

        return {
            "vector_results": list(deduped.values()),
            "graph_results": graph_results,
        }

    def _rerank_results(
        self, query: str, vector_results: List[dict], limit: int
    ) -> List[dict]:
        """Optionally rerank vector results using the configured reranker."""
        if self.reranker is None or not vector_results:
            return vector_results

        documents = [{"memory": item.get("memory", ""), **item} for item in vector_results]
        reranked = self.reranker.rerank(query, documents, top_k=limit)

        result = []
        for doc in reranked:
            item = {k: v for k, v in doc.items() if k != "memory"}
            item["memory"] = doc.get("memory", "")
            item["rerank_score"] = doc.get("rerank_score")
            result.append(item)
        return result

    def _build_search_response(self, merged_results: dict) -> dict:
        """Assemble final response dict."""
        response: dict = {"results": merged_results.get("vector_results", [])}
        if self.enable_graph:
            response["relations"] = merged_results.get("graph_results", [])
        return response

    # ------------------------------------------------------------------
    # LangGraph node wrappers (thin adapters — tested via integration)
    # ------------------------------------------------------------------

    def _node_embed(self, state: SearchState) -> dict:
        embedding = self._embed_query(state["query"])
        return {"embedding": embedding}

    def _node_vector_search(self, state: SearchState) -> dict:
        results = self._search_vector_store(
            query=state["query"],
            embeddings=state["embedding"],
            filters=state["filters"],
            limit=state["limit"],
            threshold=state["threshold"],
        )
        return {"vector_results": results}

    def _node_graph_search(self, state: SearchState) -> dict:
        results = self._search_graph(
            query=state["query"],
            filters=state["filters"],
            graph_depth=state["graph_depth"],
        )
        return {"graph_results": results}

    def _node_merge(self, state: SearchState) -> dict:
        merged = self._merge_results(
            state["vector_results"], state["graph_results"]
        )
        return {"merged_results": merged}

    def _node_rerank(self, state: SearchState) -> dict:
        reranked = self._rerank_results(
            query=state["query"],
            vector_results=state["merged_results"]["vector_results"],
            limit=state["limit"],
        )
        updated_merged = dict(state["merged_results"])
        updated_merged["vector_results"] = reranked
        return {"merged_results": updated_merged}

    def _node_build_response(self, state: SearchState) -> dict:
        response = self._build_search_response(state["merged_results"])
        return {"final_results": response}

    # ------------------------------------------------------------------
    # Conditional edge logic
    # ------------------------------------------------------------------

    def _should_search_graph(self, state: SearchState) -> str:
        if state["graph_depth"] > 0 and self.enable_graph:
            return "graph_search"
        return "merge"

    def _should_rerank(self, state: SearchState) -> str:
        if (
            state["rerank"]
            and self.reranker is not None
            and state["merged_results"].get("vector_results")
        ):
            return "rerank"
        return "build_response"
