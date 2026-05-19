import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProcessMemorySearchEngine:
    """Read-only search engine for process memory.

    Serves two flows:
    - Flow 2 (process search): search_for_step() during task execution.
    - Flow 1 (dedup search): search_for_dedup() before writing new memories.

    No LangGraph, no LLM calls, no write operations.
    All dependencies are injected via constructor.
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        graph_store: Optional[Any] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.graph_store = graph_store

    # ── Private: embedding ────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return []
        return self.embedding_model.embed(text, "search")

    # ── Private: graph search ─────────────────────────────────────────

    def _search_graph_by_brief(
        self, brief: str, filters: dict, top_k: int = 10
    ) -> list[dict]:
        if not brief or not brief.strip():
            return []
        if self.graph_store is None:
            return []
        embedding = self._embed(brief)
        if not embedding:
            return []
        return self.graph_store.search_nodes_by_embedding(
            embedding, filters, top_k=top_k
        )

    def _expand_neighbors(
        self, node_names: list[str], filters: dict, depth: int = 1
    ) -> list[dict]:
        if self.graph_store is None:
            return []
        if not node_names:
            return []
        return self.graph_store.search_nodes(node_names, filters, depth=depth)

    def _search_graph_by_step_names(
        self, step_names: list[str], filters: dict, depth: int = 10
    ) -> list[dict]:
        if self.graph_store is None:
            return []
        if not step_names:
            return []
        return self.graph_store.search_nodes(step_names, filters, depth=depth)

    def _semantic_filter(
        self,
        nodes: list[dict],
        previous_step: Optional[dict],
        threshold: float = 0.6,
    ) -> list[dict]:
        if previous_step is None:
            return nodes
        prev_brief = previous_step.get("Brief")
        if not prev_brief:
            return nodes
        if not nodes:
            return []

        prev_embedding = self._embed(prev_brief)
        if not prev_embedding:
            return nodes

        filtered = []
        for node in nodes:
            node_brief = node.get("brief")
            if not node_brief:
                continue
            node_embedding = self._embed(node_brief)
            if not node_embedding:
                continue
            similarity = self._cosine_similarity(prev_embedding, node_embedding)
            if similarity >= threshold:
                node_copy = dict(node)
                node_copy["similarity"] = similarity
                filtered.append(node_copy)

        filtered.sort(key=lambda n: n.get("similarity", 0), reverse=True)
        return filtered

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Private: vector search ────────────────────────────────────────

    def _search_chunks(
        self, goal: str, filters: dict, top_k: int = 5
    ) -> list[dict]:
        if not goal or not goal.strip():
            return []
        embedding = self._embed(goal)
        if not embedding:
            return []

        chunk_filters = {**filters, "memory_type": "process_chunk"}
        try:
            results = self.vector_store.search(
                query=goal, vectors=embedding, limit=top_k, filters=chunk_filters
            )
        except Exception as e:
            logger.error(f"_search_chunks failed: {e}")
            return []

        formatted = []
        for result in results:
            try:
                payload = getattr(result, "payload", None)
                if payload is None:
                    continue
                formatted.append({
                    "goal": payload.get("goal", ""),
                    "score": getattr(result, "score", 0.0),
                    "steps": payload.get("steps", []),
                    "id": getattr(result, "id", None),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k not in ("goal", "steps", "data", "hash", "memory_type")
                    },
                })
            except Exception:
                continue
        return formatted

    def _search_summaries(
        self, task_desc: str, filters: dict, top_k: int = 3
    ) -> list[dict]:
        if not task_desc or not task_desc.strip():
            return []
        embedding = self._embed(task_desc)
        if not embedding:
            return []

        summary_filters = {**filters, "memory_type": "process_summary"}
        try:
            results = self.vector_store.search(
                query=task_desc, vectors=embedding, limit=top_k, filters=summary_filters
            )
        except Exception as e:
            logger.error(f"_search_summaries failed: {e}")
            return []

        formatted = []
        for result in results:
            try:
                payload = getattr(result, "payload", None)
                if payload is None:
                    continue
                formatted.append({
                    "task_description": payload.get("task_description", ""),
                    "score": getattr(result, "score", 0.0),
                    "full_chain": payload.get("full_chain", []),
                    "id": getattr(result, "id", None),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k
                        not in (
                            "task_description",
                            "full_chain",
                            "data",
                            "hash",
                            "memory_type",
                        )
                    },
                })
            except Exception:
                continue
        return formatted

    # ── Public: Flow 2 — process search ───────────────────────────────

    def search_for_step(
        self,
        current_step: dict,
        previous_step: Optional[dict] = None,
        filters: Optional[dict] = None,
        task_estimate: Optional[str] = None,
        graph_hop: int = 1,
        chunk_top_k: int = 5,
        summary_top_k: int = 3,
        semantic_threshold: float = 0.6,
    ) -> dict:
        filters = filters or {}

        # Graph search
        graph_result = {"matched_nodes": [], "expanded_nodes": [], "filtered_nodes": []}

        if self.graph_store is not None and current_step.get("Brief"):
            matched = self._search_graph_by_brief(
                current_step["Brief"], filters, top_k=10
            )
            graph_result["matched_nodes"] = matched

            if matched:
                matched_names = [n["name"] for n in matched if n.get("name")]
                graph_result["expanded_nodes"] = self._expand_neighbors(
                    matched_names, filters, depth=graph_hop
                )
                graph_result["filtered_nodes"] = self._semantic_filter(
                    matched, previous_step, threshold=semantic_threshold
                )

        # Chunk search
        chunk_results = []
        if current_step.get("Goal"):
            chunk_results = self._search_chunks(
                current_step["Goal"], filters, top_k=chunk_top_k
            )

        # Summary search
        summary_results = []
        search_text = task_estimate
        if not search_text:
            parts = []
            if current_step.get("Goal"):
                parts.append(current_step["Goal"])
            if current_step.get("Brief"):
                parts.append(current_step["Brief"])
            search_text = " ".join(parts) if parts else ""
        if search_text:
            summary_results = self._search_summaries(
                search_text, filters, top_k=summary_top_k
            )

        return {
            "graph": graph_result,
            "chunks": chunk_results,
            "summaries": summary_results,
        }

    # ── Public: Flow 1 — dedup search ─────────────────────────────────

    def search_for_dedup(
        self,
        goals: list[str],
        task_description: Optional[str] = None,
        filters: Optional[dict] = None,
        chunk_top_k: int = 5,
        summary_top_k: int = 3,
    ) -> dict:
        filters = filters or {}

        # Chunk search — also collects step names for graph traversal
        all_chunks = []
        all_step_names: set[str] = set()

        for goal in goals:
            if not goal or not goal.strip():
                continue
            chunks = self._search_chunks(goal, filters, top_k=chunk_top_k)
            all_chunks.extend(chunks)
            for chunk in chunks:
                for step in chunk.get("steps", []):
                    if isinstance(step, dict) and step.get("step"):
                        all_step_names.add(step["step"])

        # Deduplicate chunks by id
        seen_ids: set[str] = set()
        deduped_chunks = []
        for chunk in all_chunks:
            cid = chunk.get("id")
            if cid is None:
                deduped_chunks.append(chunk)
            elif cid not in seen_ids:
                seen_ids.add(cid)
                deduped_chunks.append(chunk)

        # Graph search — depends on step names from chunk results
        graph_result: dict = {"chains": []}
        if self.graph_store is not None and all_step_names:
            graph_result["chains"] = self._search_graph_by_step_names(
                list(all_step_names), filters, depth=10
            )

        # Summary search — independent of chunk/graph
        summary_results = []
        search_text = task_description
        if not search_text:
            search_text = " ".join(goals) if goals else ""
        if search_text:
            summary_results = self._search_summaries(
                search_text, filters, top_k=summary_top_k
            )

        return {
            "graph": graph_result,
            "chunks": deduped_chunks,
            "summaries": summary_results,
        }
