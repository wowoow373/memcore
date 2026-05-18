# Plan: 拆分 GraphStore 与 LLM 实体提取职责（仅 Neo4j）

## Context

`mem0/memory/graph_memory.py` 中的 `MemoryGraph.search()` 同时承担了：
1. LLM 实体提取（`_retrieve_nodes_from_data` → `llm.generate_response`）
2. Cypher 图遍历（`_search_graph_db_by_depth`）
3. BM25 重排序

这导致 GraphStore 层耦合了 NLP、图存储、检索三个领域，调用方无法跳过 LLM 提取或替换提取策略。

本计划只修改 Neo4j 实现（`graph_memory.py`），其他图数据库（memgraph、kuzu、apache_age、neptune）不改动，通过 SearchEngine 层的适配逻辑保持兼容。

## Constraints

- 只改 `mem0/memory/graph_memory.py`（Neo4j 实现）
- 其他图数据库文件不动
- `Memory.add()` 流程不动（仍需 LLM 提取关系）
- 旧的 `MemoryGraph.search()` 接口保留做 backward-compatible
- 现有测试 `tests/memory/test_neo4j_cypher_syntax.py` 继续通过

## Implementation

### Step 1: `graph_memory.py` — 新增纯图遍历接口 `search_nodes()`

新增方法签名：

```python
def search_nodes(self, node_names: list[str], filters, depth: int = 2, limit: int = 100):
    """Pure graph traversal from given node names.

    Args:
        node_names: Pre-extracted node names to start traversal from.
        filters: Scope filters (user_id, agent_id, run_id).
        depth: Traversal depth (hops).
        limit: Max number of relations to return.

    Returns:
        list[dict]: Reranked relations with keys source/relationship/destination.
    """
```

实现：从现有 `search()` 中抽取以下逻辑，**移除 LLM 提取部分**：
- 节点名 normalize（lower + space→underscore）
- `_search_graph_db_by_depth()` 调用
- 移除BM25 rerank

### Step 2: `graph_memory.py` — 简化 `search()` 为兼容层

修改现有 `search(self, query, filters, limit=100)`：

```python
def search(self, query, filters, limit=100):
    """Backward-compatible wrapper. No LLM extraction.

    - query is list/tuple/set: pass directly to search_nodes.
    - query is str: split by comma (or keep as single item) → search_nodes.
    """
```

具体改动：
- 删除内部对 `_retrieve_nodes_from_data()` 的调用
- 删除 TODO 注释中的"职责边界混乱"段落（问题解决后移除）
- `__init__` 中保留 `self.llm`（`add()` 仍需要）

### Step 3: `search_engine.py` — 上移实体提取到 SearchEngine

修改 `SearchEngine.__init__`：

```python
def __init__(self, embedding_model, vector_store, graph_store=None, reranker=None, llm=None):
    # ... existing ...
    self.llm = llm
```

新增 `_extract_nodes(self, query: str, filters: dict) -> list[str]`：
- 从 `graph_memory.py` 的 `_retrieve_nodes_from_data()` 迁移 LLM tool-call 提取逻辑
- 保留 self-reference 处理（"I"/"me"/"my" → user_id）
- 保留 normalize（lower + space→underscore）

修改 `_search_graph(self, query, filters, graph_depth)`：

```python
def _search_graph(self, query: str, filters: dict, graph_depth: int) -> List[dict]:
    if graph_depth <= 0 or not self.enable_graph:
        return []

    # Prefer new search_nodes interface (Neo4j)
    if hasattr(self.graph, 'search_nodes'):
        if self.llm is not None:
            node_names = self._extract_nodes(query, filters)
        else:
            # Fallback: simple split when no LLM available
            node_names = [n.strip() for n in query.split(",") if n.strip()] if "," in query else [query.strip()]
        return self.graph.search_nodes(node_names, filters, depth=graph_depth)

    # Legacy fallback for other graph stores
    return self.graph.search(query, filters, limit=graph_depth)
```

修改 `_node_graph_search(self, state)` 以适配。

### Step 4: `main.py` — 将 LLM 实例传入 SearchEngine

在 `Memory.__init__` 中：

```python
self.search_engine = SearchEngine(
    embedding_model=self.embedding_model,
    vector_store=self.vector_store,
    graph_store=self.graph if self.enable_graph else None,
    reranker=self.reranker,
    llm=self.llm,  # ← 新增
)
```

在 `AsyncMemory.__init__` 中同样处理。

## Verification

1. **单元测试**: 运行 `pytest tests/memory/test_neo4j_cypher_syntax.py -v`，确认全部通过
2. **Mock 验证**: `MemoryGraph.search_nodes()` 接收 `list[str]`，内部不访问 `self.llm`
3. **接口兼容**: 直接调用 `graph.search("alice, bob", filters)` 仍能工作（走兼容层）
4. **集成验证**: `Memory.search(query="...", user_id="u1")` 仍能返回 `relations` 字段

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `mem0/memory/graph_memory.py` | 103-212 | 拆分 `search()` → 新增 `search_nodes()` + 简化 `search()` 为兼容层 |
| `mem0/memory/graph_memory.py` | 132-153 | 移除已解决的 TODO 注释 |
| `mem0/memory/search_engine.py` | 47-58 | `__init__` 新增 `llm` 参数 |
| `mem0/memory/search_engine.py` | 202-208 | `_search_graph()` 增加 `search_nodes` 适配 + `_extract_nodes()` |
| `mem0/memory/main.py` | 306-313 | `SearchEngine` 初始化传入 `llm=self.llm` |
| `mem0/memory/main.py` | ~1294 附近 | `AsyncMemory` 同样传入 `llm` |
