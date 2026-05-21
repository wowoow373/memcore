# ProcessMemorySearchEngine 详细设计

## 一、概述

ProcessMemorySearchEngine 是流程记忆的**纯只读**检索引擎。服务两个场景：Flow 1 去重搜索（Add 前查重）和 Flow 2 过程搜索（执行中联想）。

不涉及 LangGraph —— 检索链路是线性的三层并行调用，无需状态机编排。对外两个公开入口，内部分别组合私有方法。

---

## 二、初始化与依赖

```python
class ProcessMemorySearchEngine:
    def __init__(
        self,
        embedding_model,   # EmbedderFactory 创建的实例，提供 embed(text, action) -> list[float]
        vector_store,      # VectorStoreBase 子类实例（Chunk + Summary 共用）
        graph_store,       # MemoryGraph 实例（node_label 配置为 Step 专用）
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.graph_store = graph_store
```

三个依赖全部外部注入，Engine 内部不创建任何组件。`graph_store` 需要预先配置好 Step 节点所用的 `node_label`（与 Entity 节点标签隔离）。

---

## 三、公开接口

### 3.1 Flow 2：过程检索

```python
def search_for_step(
    self,
    current_step: dict,            # 当前 step 的 summary {"Goal","Step","Action","Dependency","Brief"}
    previous_step: dict | None,    # 前一步的 summary，None 表示无前一步
    filters: dict,                 # {"user_id": str, "agent_id": str|None, "run_id": str|None}
    task_estimate: str | None = None,  # 任务类型估计，None 时由 current_step["Goal"] 拼接生成
    graph_hop: int = 1,            # 图邻居扩展跳数
    chunk_top_k: int = 5,          # Chunk 召回数量
    summary_top_k: int = 3,        # Summary 召回数量
    semantic_threshold: float = 0.6,  # 前一步语义筛选阈值
) -> dict:
```

**内部步骤**：

1. **图搜索**：`_search_graph_by_brief(current_step["Brief"], filters)` → 匹配 N 个 Step 节点
2. **图扩展**：`_expand_neighbors(matched_node_names, filters, graph_hop)` → 1-hop 邻居
3. **语义再筛选**：若 `previous_step` 不为 None，调 `_semantic_filter(expanded_nodes, previous_step, semantic_threshold)` 筛选
4. **Chunk 搜索**：`_search_chunks(current_step["Goal"], filters, chunk_top_k)` → 相似 Goal 的 chunk
5. **Summary 搜索**：`_search_summaries(task_estimate or _build_task_estimate(current_step), filters, summary_top_k)` → 相似完整链路

**返回**：

```python
{
    "graph": {
        "matched_nodes": [...],    # Brief 匹配到的 Step 节点
        "expanded_nodes": [...],   # 1-hop 邻居节点
        "filtered_nodes": [...],   # 前一步语义筛选后的节点（previous_step=None 时同 expanded_nodes）
    },
    "chunks": [
        {
            "goal": str,
            "score": float,
            "steps": [{"step": str, "brief": str, "action": str, "dependency": [...]}, ...],
            "id": str | None,      # 向量库中的记录 ID
            "metadata": dict,
        }
    ],
    "summaries": [
        {
            "task_description": str,
            "score": float,
            "full_chain": [{"step": str, "brief": str, "goal": str, "dependency": [...]}, ...],
            "id": str | None,
            "metadata": dict,
        }
    ],
}
```

### 3.2 Flow 1：去重检索

```python
def search_for_dedup(
    self,
    goals: list[str],               # 从 summaries 提取的去重 Goal 列表
    task_description: str | None,   # 任务宏观描述（可为 None，此时用 goals 拼接）
    filters: dict,
    chunk_top_k: int = 5,
    summary_top_k: int = 3,
) -> dict:
```

**内部步骤**：

1. **Chunk 搜索**（先执行）：对每个 goal 调 `_search_chunks(goal, filters, chunk_top_k)` → 已有相似 chunk → 从 chunk 结果中提取已有 step 名称
2. **Graph 搜索**（依赖 Chunk 结果）：用 Chunk 返回的 step 名称集合，调 `_search_graph_by_step_names(step_names, filters)` → `MemoryGraph.search_nodes(step_names, depth=10)` 遍历完整链路。不新增 MemoryGraph 按属性查询的方法
3. **Summary 搜索**（独立，与 Chunk 并行）：对 `task_description` 调 `_search_summaries(task_desc, filters, summary_top_k)`

**返回**：与 `search_for_step` 结构一致，但 `graph` 中 `matched_nodes` 替换为 `chains`（从已有 step 名称出发遍历的完整链路）。

---

## 四、私有方法 —— 图搜索

### _search_graph_by_brief

```python
def _search_graph_by_brief(
    self, brief: str, filters: dict, top_k: int = 10
) -> list[dict]:
```

**逻辑**：
1. `embedding = self.embedding_model.embed(brief, "search")`
2. 调 `self.graph_store.search_nodes_by_embedding(embedding, filters, top_k)`
3. 返回 `[{"name": str, "brief": str, "goal": str, "step": str, "score": float, ...}]`

**依赖**：`MemoryGraph` 新增方法 `search_nodes_by_embedding`（见第五节）。

### _expand_neighbors

```python
def _expand_neighbors(
    self, node_names: list[str], filters: dict, depth: int = 1
) -> list[dict]:
```

**逻辑**：
1. 调 `self.graph_store.search_nodes(node_names, filters, depth=depth)`
2. 返回 `[{"source": str, "relationship": str, "destination": str}, ...]`

**边界**：纯遍历，不做语义匹配。`search_nodes` 已经是 `MemoryGraph` 的现有接口。

### _semantic_filter

```python
def _semantic_filter(
    self, nodes: list[dict], previous_step: dict, threshold: float = 0.6
) -> list[dict]:
```

**逻辑**：
1. `prev_embedding = self.embedding_model.embed(previous_step["Brief"], "search")`
2. 对每个候选节点的 Brief 做 embedding，计算余弦相似度
3. 保留 `cosine(prev_embedding, node_brief_embedding) >= threshold` 的节点
4. 按相似度降序排列返回

**边界**：这是一个纯 Python 计算，不涉及外部 I/O。如果候选节点数量大，可以在写入时预存 Brief embedding 到节点属性，直接在 Cypher 里筛。但初期候选量小（1-hop 后通常 < 20），本地计算即可。

### _search_graph_by_step_names

```python
def _search_graph_by_step_names(
    self, step_names: list[str], filters: dict, depth: int = 10
) -> list[dict]:
```

**逻辑**：
1. 接收从 Chunk 召回中提取的已有 step 名称列表
2. 调 `self.graph_store.search_nodes(step_names, filters, depth=depth)` 沿 DEPENDS_ON 遍历
3. 返回 `[{"source": str, "relationship": str, "destination": str}, ...]`

**边界**：仅用于 Flow 1 去重搜索。不新增 MemoryGraph 方法，直接复用 `search_nodes`（按节点名精准匹配 + 遍历）。step 名称从 Chunk 召回结果的 `steps` 字段中提取。

---

## 五、MemoryGraph 新增方法

在 `mem0/memory/graph_memory.py` 的 `MemoryGraph` 类中新增：

```python
def search_nodes_by_embedding(
    self,
    embedding: list[float],    # 外部已计算好的 embedding，本方法不做 embed
    filters: dict,             # 至少含 user_id
    top_k: int = 10,
    threshold: float = 0.6,
) -> list[dict]:
    """按 embedding 余弦相似度查找语义最接近的节点。

    在 Step 节点上使用，匹配 n.brief_embedding 与传入的 embedding。
    调用方（ProcessMemorySearchEngine）负责预先计算 embedding，本方法只做 Cypher 查询。
    """
```

**Cypher 实现**（`node_label` 根据 graph_store 配置）：

```cypher
MATCH (n {node_label} {user_id: $user_id})
WHERE n.brief_embedding IS NOT NULL
WITH n, vector.similarity.cosine(n.brief_embedding, $embedding) AS score
WHERE score >= $threshold
RETURN n.name AS name, n.brief AS brief, n.goal AS goal, n.step AS step, score
ORDER BY score DESC
LIMIT $top_k
```

---

## 六、私有方法 —— 向量搜索

### _search_chunks

```python
def _search_chunks(
    self, goal: str, filters: dict, top_k: int = 5
) -> list[dict]:
```

**逻辑**：
1. `embedding = self.embedding_model.embed(goal, "search")`
2. `results = self.vector_store.search(query=goal, vectors=embedding, limit=top_k, filters=filters)`
   - filters 中附加 `{"memory_type": "process_chunk"}` 确保只召回 chunk
3. 格式化返回 `[{"goal", "score", "steps", "id", "metadata"}, ...]`

### _search_summaries

```python
def _search_summaries(
    self, task_desc: str, filters: dict, top_k: int = 3
) -> list[dict]:
```

**逻辑**：
1. `embedding = self.embedding_model.embed(task_desc, "search")`
2. `results = self.vector_store.search(query=task_desc, vectors=embedding, limit=top_k, filters=filters)`
   - filters 中附加 `{"memory_type": "process_summary"}` 确保只召回 summary
3. 格式化返回 `[{"task_description", "score", "full_chain", "id", "metadata"}, ...]`

---

## 七、节点职责边界总结

| 方法 | 职责 | 外部调用 | LLM 调用 | 写操作 |
|------|------|---------|---------|--------|
| `search_for_step` | Flow 2 入口，编排三层并行搜索 + 语义筛选 | 无 | 无 | 无 |
| `search_for_dedup` | Flow 1 入口，编排三层并行搜索 | 无 | 无 | 无 |
| `_search_graph_by_brief` | Brief embedding → MemoryGraph.search_nodes_by_embedding | `MemoryGraph` | 无 | 无 |
| `_expand_neighbors` | 节点名 → MemoryGraph.search_nodes 遍历 | `MemoryGraph` | 无 | 无 |
| `_semantic_filter` | 前一步 Brief × 候选节点 Brief 余弦筛选 | 无（纯计算） | 无 | 无 |
| `_search_graph_by_step_names` | Chunk 返回的 step 名 → MemoryGraph.search_nodes 全链路遍历 | `MemoryGraph` | 无 | 无 |
| `_search_chunks` | Goal → vector_store.search | `vector_store` | 无 | 无 |
| `_search_summaries` | task_desc → vector_store.search | `vector_store` | 无 | 无 |

**无 LLM，无写入。**
