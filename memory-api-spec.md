# Memory 模块接口规格书

## 一、架构总览

```
                           Memory (main.py)
                          /                \
        标准记忆路径                           流程记忆路径
   ┌──────┴──────┐                    ┌───────┴───────┐
   │  SearchEngine │ ← 只读召回        │ ProcessMemory │
   │  AddEngine    │ ← 写入编排        │ SearchEngine  │ ← 只读召回
   └──────────────┘                    │ AddEngine     │ ← 写入编排
                                       └───────────────┘
```

**核心重构**：将 `Memory.add()` 和 `Memory.search()` 中的内联逻辑拆分为独立的 Engine 类，每个 Engine 内部用 LangGraph 编排步骤。流程记忆（Process Memory）复用相同的 `Search → Decide → Execute` 模式，但输入/输出数据结构不同。

**核心语义 "add-with-search-back"**：Add 在写入前必须调用 Search 召回已有记忆，基于召回结果做去重决策，并将召回的记忆随写入结果一起返回给调用方。

---

## 二、目录结构

```
mem0/memory/
├── main.py                         # Memory + AsyncMemory：接口层、初始化各 Engine、委托调用
├── base.py                         # MemoryBase（不动）
├── storage.py                      # SQLiteManager：history 记录（不动）
├── telemetry.py                    # 遥测（不动）
├── utils.py                        # 通用工具（不动）
├── graph_memory.py                 # MemoryGraph：标准记忆图存储（复用 ingest/search_nodes 等非 LLM 接口）
├── search_engine.py                # 【新增】SearchEngine：标准记忆统一召回
├── add_engine.py                   # 【新增】AddEngine：标准记忆写入编排
├── process_search_engine.py        # 【新增】ProcessMemorySearchEngine：流程记忆统一召回
├── process_add_engine.py           # 【新增】ProcessMemoryAddEngine：流程记忆写入编排
├── memgraph_memory.py              # 已有
├── kuzu_memory.py                  # 已有
├── apache_age_memory.py            # 已有
└── setup.py                        # 已有
```

### 各文件职责

| 文件 | 职责 | LangGraph | LLM 调用 | 写操作 |
|------|------|-----------|---------|--------|
| `search_engine.py` | 标准记忆统一召回：向量 + 图 + 合并 + rerank | 有（6 节点 2 条件分支） | 仅在图搜索时调用 LLM 提取实体 | 无 |
| `add_engine.py` | 标准记忆写入编排：预处理 → 搜索 → 决策 → 执行向量 → 提取图 → 执行图 | 有（8 节点 2 条件分支） | 3 次（提取搜索查询 + 决策 ADD/UPDATE/DELETE + 提取图实体关系） | 有 |
| `process_search_engine.py` | 流程记忆统一召回：图 Brief 语义匹配 + Chunk/Summary 向量召回 | 无（纯函数调用） | 无 | 无 |
| `process_add_engine.py` | 流程记忆写入编排：preprocess → search → decide → execute | 有（5 节点 0 条件分支） | 1 次（统一决策三层 ADD/UPDATE/MERGE/NONE） | 有 |

---

## 三、配置

### 3.1 `MemoryConfig` 新增字段

在 [mem0/configs/base.py](mem0/configs/base.py#L67-L70)：

```python
class MemoryConfig(BaseModel):
    # ... 已有字段 ...
    process_memory: Optional[ProcessMemoryConfig] = Field(
        description="Configuration for process memory (step-level task memory)",
        default=None,
    )
```

### 3.2 `ProcessMemoryConfig`

[mem0/configs/base.py](mem0/configs/base.py#L73-L99)：

```python
class ProcessMemoryConfig(BaseModel):
    vector_store: VectorStoreConfig       # Chunk + Summary 的向量存储配置
    graph_store: GraphStoreConfig         # Step 节点的 Neo4j 配置
    graph_search_depth: int = 10          # Flow 1 去重时图遍历深度（默认 10）
    chunk_top_k: int = 5                  # Chunk 召回数量
    summary_top_k: int = 3                # Summary 召回数量
    semantic_filter_threshold: float = 0.6  # 前一步语义筛选余弦相似度阈值 [0.0, 1.0]
```

当 `process_memory` 为 `None` 时，流程记忆功能关闭。设置为 `ProcessMemoryConfig(...)` 后，`Memory.__init__()` 会初始化：
- `self.process_vector_store` — 专用向量库
- `self.process_graph_store` — `MemoryGraph` 实例（`node_label=":Step"`）
- `self.process_search_engine` — `ProcessMemorySearchEngine` 实例
- `self.process_add_engine` — `ProcessMemoryAddEngine` 实例

---

## 四、标准记忆路径

### 4.1 SearchEngine — 统一召回

文件：[mem0/memory/search_engine.py](mem0/memory/search_engine.py)

**职责**：纯只读的"给定 query → 返回所有相关记忆"。向量相似度召回 + 图关系遍历召回 + 合并去重 + 可选 rerank。

**LangGraph 图结构**：

```
START → embed → vector_search ──┬──(graph_depth>0?)──→ graph_search → merge
                                 │                                        │
                                 └──(graph_depth=0)──→ merge ←─────────────┘
                                                         │
                                            (rerank=True?)──→ rerank → build_response → END
                                                         │
                                            (rerank=False)─→ build_response → END
```

**6 个节点、2 个条件分支**。对外暴露的 `search()` 方法直接调用编译后的图。

**公开接口**：

```python
search(
    query: str,
    filters: dict,
    limit: int = 100,
    threshold: Optional[float] = None,
    graph_depth: int = 2,
    rerank: bool = True,
) -> dict
```

**返回**：

```python
# 有图存储：
{
    "results": [
        {
            "id": "uuid",
            "memory": "Name is John",
            "hash": "md5",
            "score": 0.92,
            "created_at": "2024-01-15T10:30:00+00:00",
            "updated_at": "...",
            "metadata": {"user_id": "u1", ...}
        }
    ],
    "relations": [
        {"source": "John", "relationship": "lives_in", "destination": "NYC"}
    ]
}

# 无图存储：
{"results": [...]}
```

**SearchState**：

```python
class SearchState(TypedDict):
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
```

---

### 4.2 AddEngine — 写入编排（add-with-search-back）

文件：[mem0/memory/add_engine.py](mem0/memory/add_engine.py)

**职责**：接收对话消息，编排写入流程。核心模式：**先 Search 召回已有记忆 → LLM 对比决策 → 执行写入 → 返回操作结果 + 回忆起的记忆**。

**add-with-search-back 含义**：Add 在写入之前显式调用 SearchEngine 获取相关已有记忆，基于此做 ADD/UPDATE/DELETE/NONE 决策。Search 返回的记忆原样放入 Add 的返回结果中（`recalled_memories` 字段），让调用方（外部 Agent）感知到系统联想起了哪些记忆。

**LangGraph 图结构**：

```
START → preprocess ─┬──(infer=True)──→ extract_queries → search → decide_memory → execute_vector
                     │                                                                  │
                     └──(infer=False)─→ direct_add ──────────────────────────────────────┤
                                                                                         │
                                           ┌─────────────────────────────────────────────┘
                                           │
                                           ├──(enable_graph=True)──→ extract_graph → execute_graph → assemble_result → END
                                           │
                                           └──(enable_graph=False)─→ assemble_result → END
```

**8 个节点、2 个条件分支**。

| 节点 | 职责 | LLM 调用 | 写操作 |
|------|------|---------|--------|
| `preprocess` | 拼接 messages 为纯文本 | 无 | 无 |
| `direct_add` | infer=False 快速路径：逐条 message 直接存向量库 | 无 | 有 |
| `extract_queries` | LLM 从对话中提取搜索查询列表 | **1 次** | 无 |
| `search` | → 调 SearchEngine.search() 统一召回 | SearchEngine 内部 | 无 |
| `decide_memory` | LLM 对比新对话 + 召回结果 → ADD/UPDATE/DELETE/NONE | **1 次** | 无 |
| `execute_vector` | 按决策操作向量库（insert/update/delete） | 无 | 有 |
| `extract_graph` | LLM 从对话中提取实体/关系/待删除关系 | **1 次** | 无 |
| `execute_graph` | → 调 MemoryGraph.ingest() 写入图 | 无 | 有 |
| `assemble_result` | 合并 results + recalled_memories + relations | 无 | 无 |

**公开接口**：

```python
def add(
    self,
    messages: list[dict],        # 已验证、归一化的消息列表
    metadata: dict,              # 含 user_id/agent_id/run_id
    filters: dict,               # 查询筛选条件
    infer: bool = True,          # False = 直接存，不走 LLM
) -> dict:
```

**AddState**：

```python
class AddState(TypedDict):
    messages: list[dict]
    metadata: dict
    filters: dict
    infer: bool
    parsed_messages: str
    search_queries: list[str]
    recalled_memories: dict       # {"results": [MemoryItem], "relations": [...]}
    decisions: list[dict]         # [{"id": str|null, "text": str, "event": str, "old_memory": str}]
    entity_type_map: dict         # {"entity": "type"}
    relations: list[dict]         # [{"source", "relationship", "destination"}]
    to_be_deleted: list[dict]
    graph_result: dict
    results: list[dict]
    final_results: dict
    error: Optional[str]
```

**返回 — add-with-search-back 关键结构**：

```python
{
    # 实际执行的操作
    "results": [
        {"id": "uuid-new", "memory": "Name is John", "event": "ADD"},
        {"id": "uuid-old", "memory": "Likes tennis on weekends", "event": "UPDATE", "previous_memory": "Likes tennis"},
        {"id": "uuid-del", "memory": "Dislikes pizza", "event": "DELETE"},
    ],
    # ← 这是 "add-with-search-back" 的关键：Search 召回的记忆原样返回
    "recalled_memories": {
        "results": [
            {"id": "uuid-old", "memory": "Likes tennis", "score": 0.85, "metadata": {...}},
        ],
        "relations": [
            {"source": "John", "relationship": "lives_in", "destination": "NYC"}
        ]
    },
    # 图存储结果（若启用）
    "relations": {
        "deleted_entities": [...],
        "added_entities": [...]
    }
}
```

---

### 4.3 `Memory.add()` — 路由逻辑

文件：[mem0/memory/main.py](mem0/memory/main.py#L412-L534)

```python
def add(self, messages, *, user_id=None, agent_id=None, run_id=None,
        metadata=None, infer=True, memory_type=None, prompt=None) -> dict:
```

路由规则：

| `memory_type` | 条件 | 路由 |
|---------------|------|------|
| `None` (默认) | — | `self.add_engine.add(messages, metadata, filters, infer)` |
| `"procedural_memory"` | `agent_id` 存在 | `self._create_procedural_memory(messages, metadata, prompt)`（仍为旧路径） |
| `"process_memory"` | `config.process_memory` 已配置 | `self.process_add_engine.add(summaries=messages, metadata=..., filters=...)` |
| `"process_memory"` | `config.process_memory` 未配置 | `Mem0ValidationError` (VALIDATION_005) |

---

### 4.4 `Memory.search()` — 委托给 SearchEngine

文件：[mem0/memory/main.py](mem0/memory/main.py#L684-L765)

```python
def search(self, query, *, user_id=None, agent_id=None, run_id=None,
           limit=100, filters=None, threshold=None, rerank=True) -> dict:
```

内部调用 `self.search_engine.search(query, effective_filters, limit, threshold, graph_depth=2, rerank)`。

**关键**：`graph_depth` 当前硬编码为 2，暂未在 `Memory.search()` 签名中暴露。待后续配置化时通过 `MemoryConfig.graph_search_depth` 暴露。

---

### 4.5 `MemoryGraph.ingest()` — 图写入统一接口

文件：[mem0/memory/graph_memory.py](mem0/memory/graph_memory.py#L888-L1028)

**签名**：

```python
def ingest(
    self,
    entity_type_map: dict,                         # {entity_name: entity_type}
    relations: list[dict],                          # [{"source", "relationship", "destination"}]
    filters: dict,                                  # {user_id, agent_id?, run_id?}
    to_be_deleted: Optional[list[dict]] = None,     # 待删除的关系
    node_properties: Optional[dict[str, dict]] = None,  # 流程记忆扩展：{node_name: {brief, goal, action, brief_embedding}}
) -> dict:
```

**与 `MemoryGraph.add()` 的区别**：`ingest()` 不调用 LLM，只做纯 MERGE/CREATE/ DELETE 操作。实体提取、关系提取、删除决策全部由上游（AddEngine / ProcessMemoryAddEngine）完成。

兼容性：`node_properties=None` 时行为与标准记忆完全一致，只 SET mentions。

---

### 4.6 `MemoryGraph.search_nodes_by_embedding()` — 语义节点查找

文件：[mem0/memory/graph_memory.py](mem0/memory/graph_memory.py#L153-L218)

用于流程记忆 Flow 2 的图搜索。不查看节点的 embedding 属性，而是查看 `brief_embedding` 向量属性。

```python
def search_nodes_by_embedding(
    self,
    embedding: list[float],     # 外部已计算好的 embedding
    filters: dict,              # 至少含 user_id
    top_k: int = 10,
    threshold: float = 0.6,
) -> list[dict]:
```

Cypher 查询：
```cypher
MATCH (n {node_label} {user_id: $user_id})
WHERE n.brief_embedding IS NOT NULL
WITH n, vector.similarity.cosine(n.brief_embedding, $embedding) AS score
WHERE score >= $threshold
RETURN n.name AS name, n.brief AS brief, n.goal AS goal,
       n.step AS step, n.action AS action, score
ORDER BY score DESC LIMIT $top_k
```

---

## 五、流程记忆路径

### 5.1 两层流程

| 流程 | 时机 | 操作 | 引擎入口 |
|------|------|------|---------|
| Flow 1 | 任务完成后 | Search（去重） → Add/Update | `ProcessMemoryAddEngine.add()` |
| Flow 2 | 任务进行中 | 仅 Search，不写 | `ProcessMemorySearchEngine.search_for_step()` |

### 5.2 三层记忆颗粒度

| 层 | 存储 | 粒度 | 回答 | Flow 1 写入方式 | Flow 2 检索方式 |
|----|------|------|------|----------------|----------------|
| **Graph** | Neo4j `(:Step)-[:DEPENDS_ON]->(:Step)` | 细 — 单步 | "做这步前/后需要什么" | `MemoryGraph.ingest(relations, node_properties={name: {brief,goal,action,brief_embedding}})` | Brief embedding 语义匹配 → 1-hop 扩展 → 前一步语义筛选 |
| **Chunk** | 向量库 `memory_type="process_chunk"` | 中 — 子目标 | "这个子目标怎么做" | `vector_store.insert/update(goal, steps)` ; 同 Goal MERGE | Goal 向量召回 |
| **Summary** | 向量库 `memory_type="process_summary"` | 粗 — 完整任务 | "整个任务长什么样" | `vector_store.insert/update(task_description, full_chain)` | 任务描述向量召回 |

### 5.3 流程记忆输入格式

外部 Code Agent 每一步产出的结构化 summary：

```json
{
    "Goal": "Add user authentication",
    "Step": "03 - Create auth.py",
    "Action": "create_file(path='auth.py')",
    "Dependency": [
        {"step_id": "01 - Read main.py", "description": "Parse entry logic"}
    ],
    "Brief": "Create auth.py and implement login/logout functions"
}
```

一次任务完成交付完整 summary 数组。

---

### 5.4 ProcessMemorySearchEngine — 流程记忆统一召回

文件：[mem0/memory/process_search_engine.py](mem0/memory/process_search_engine.py)

纯只读。无 LangGraph，无 LLM 调用。两个公开入口。

#### `search_for_dedup()` — Flow 1 去重检索

```python
def search_for_dedup(
    self,
    goals: list[str],
    task_description: Optional[str] = None,
    filters: Optional[dict] = None,
    chunk_top_k: int = 5,
    summary_top_k: int = 3,
) -> dict:
```

**内部步骤**：
1. **Chunk 搜索（先执行）**：对每个 goal → `_search_chunks(goal)` → 向量库（memory_type="process_chunk"）→ 提取已有 step 名称
2. **Graph 搜索（依赖 Chunk 结果）**：用 Chunk 返回的 step 名称 → `MemoryGraph.search_nodes(step_names, depth=10)` → 遍历完整 DEPENDS_ON 链
3. **Summary 搜索（独立）**：对 task_description（或 goals 拼接） → `_search_summaries(task_desc)` → 向量库（memory_type="process_summary"）

**返回**：

```python
{
    "graph": {
        "chains": [
            {"source": "01 - Read main.py", "relationship": "DEPENDS_ON", "destination": "03 - Create auth.py"},
        ]
    },
    "chunks": [
        {"goal": "Add user authentication", "score": 0.85, "steps": [...], "id": "chunk-uuid", "metadata": {...}},
    ],
    "summaries": [
        {"task_description": "Implement user auth system", "score": 0.78, "full_chain": [...], "id": "summary-uuid", "metadata": {...}},
    ]
}
```

#### `search_for_step()` — Flow 2 过程检索

```python
def search_for_step(
    self,
    current_step: dict,                      # {"Goal", "Step", "Action", "Brief"}
    previous_step: Optional[dict] = None,    # 前一步，用于语义再筛选
    filters: Optional[dict] = None,
    task_estimate: Optional[str] = None,     # 任务类型估计，None 时用 Goal+Brief 拼接
    graph_hop: int = 1,
    chunk_top_k: int = 5,
    summary_top_k: int = 3,
    semantic_threshold: float = 0.6,
) -> dict:
```

**内部步骤**：

1. **图搜索**：
   - `_search_graph_by_brief(current_step["Brief"])` → embed Brief → `MemoryGraph.search_nodes_by_embedding()` 语义匹配
   - `_expand_neighbors(matched_names, depth=graph_hop)` → `MemoryGraph.search_nodes(names, depth)` 1-hop 扩展
   - 若 `previous_step` 非空：`_semantic_filter(matched_nodes, previous_step["Brief"], threshold)` → 计算 cosine 相似度筛选

2. **Chunk 搜索**（独立）：`_search_chunks(current_step["Goal"])` → 向量库

3. **Summary 搜索**（独立）：`_search_summaries(task_estimate || Goal+Brief)` → 向量库

**返回**：

```python
{
    "graph": {
        "matched_nodes": [
            {"name": "03 - Create auth.py", "brief": "...", "goal": "...", "step": "03 - Create auth.py", "action": "...", "score": 0.92}
        ],
        "expanded_nodes": [
            {"source": "01", "relationship": "DEPENDS_ON", "destination": "03"}
        ],
        "filtered_nodes": [...]     # previous_step 语义筛选后
    },
    "chunks": [...],
    "summaries": [...]
}
```

---

### 5.5 ProcessMemoryAddEngine — 流程记忆写入编排

文件：[mem0/memory/process_add_engine.py](mem0/memory/process_add_engine.py)

仅 Flow 1 使用。LangGraph 编排 5 个节点，0 个条件分支（流程记忆始终走全流程）。

**LangGraph 图**：

```
START → preprocess → search → decide → execute → assemble → END
```

| 节点 | 职责 | LLM 调用 | 写操作 |
|------|------|---------|--------|
| `preprocess` | 解析 summaries → goals, steps, dependencies, entity_type_map, task_description | 无 | 无 |
| `search` | → `ProcessMemorySearchEngine.search_for_dedup()` 三层去重召回 | 无 | 无 |
| `decide` | 一次 LLM：新 summaries + 三层 recall → 三层各自的 ADD/UPDATE/MERGE/NONE | **1 次** | 无 |
| `execute` | 三层独立写入：Graph 调 `MemoryGraph.ingest(node_properties=...)` ，Chunk 和 Summary 调 `vector_store.insert/update` | 无 | **有** |
| `assemble` | 合并 results + recalled → `{results: {...}, recalled: {...}}` | 无 | 无 |

**公开接口**：

```python
def add(
    self,
    summaries: list[dict],      # 完整 summary 数组
    metadata: dict,             # {user_id, agent_id?, run_id?}
    filters: dict,              # 查询筛选条件
) -> dict:
```

**返回**：

```python
{
    "results": {
        "graph": {
            "deleted_entities": [...],
            "added_entities": [...]
        },
        "chunks": [
            {"id": "chunk-uuid", "goal": "Add user auth", "event": "ADD"}  # ADD|MERGE|UPDATE
        ],
        "summary": {
            "id": "summary-uuid", "event": "ADD", "task_description": "Implement..."
        }
    },
    "recalled": {
        "graph": {"chains": [...]},
        "chunks": [...],
        "summaries": [...]
    }
}
```

**决策 LLM Prompt 输出格式**：

```json
{
  "graph": {
    "steps": [
      {"name": "03 - Create auth.py", "event": "ADD", "goal": "Add user auth", "brief": "...", "action": "create_file()"}
    ],
    "edges": [
      {"source": "01 - Read main.py", "target": "03 - Create auth.py", "relationship": "DEPENDS_ON", "event": "ADD"}
    ]
  },
  "chunks": [
    {"goal": "Add user auth", "event": "MERGE", "merge_with": "<existing_id>", "steps": [...]}
  ],
  "summary": {
    "event": "ADD",
    "task_description": "Implement complete user authentication system",
    "full_chain": [{"step": "...", "brief": "..."}]
  }
}
```

**Decision 校验规则**（与 AddEngine.decide_memory 一致）：

- ADD event：强制 `id = null`
- UPDATE/NONE：id 必须在 recalled 中存在，否则降级为 ADD
- MERGE（chunk）：`merge_with` 必须在 recalled chunk ids 中存在，否则降级为 ADD
- recalled 为空 → 所有变为 ADD

---

## 六、"add-with-search-back" 完整数据流

### 6.1 标准记忆路径

```
外部调用 Memory.add(messages, ...)
    │
    ▼
AddEngine.add()
    │
    ├─ preprocess       : messages → parsed_messages 纯文本
    │
    ├─ extract_queries  : LLM(parsed_messages) → search_queries[]
    │                                                            ┌──────────────────┐
    ├─ search           : SearchEngine.search(query, filters) ── │ ← 这是 "search" │
    │                    │  └─ vector_store.search()              │   Add 调用       │
    │                    │  └─ graph.search_nodes() (可选)        │   Search 召回    │
    │                    │  └─ reranker.rerank() (可选)           │   已有记忆       │
    │                    → recalled_memories = {results[], relations[]}              │
    │                                                            └──────────────────┘
    ├─ decide_memory    : LLM(parsed_messages + recalled_memories)
    │                    → decisions[] (ADD/UPDATE/DELETE/NONE)
    │
    ├─ execute_vector   : 按 decisions 操作 vector_store
    │                    → results[] (实际写入结果)
    │
    ├─ extract_graph    : LLM(parsed_messages + existing_relations)
    │                    → entity_type_map, relations, to_be_deleted
    │
    ├─ execute_graph    : graph.ingest(entity_type_map, relations, filters, to_be_deleted)
    │                    → graph_result
    │                                               ┌────────────────────────────────┐
    └─ assemble_result  : final = {               │ ← 这是 "back"                  │
         "results": results[],                      │   写入结果 + 召回记忆          │
         "recalled_memories": recalled_memories,    │   一起返回给调用方             │
         "relations": graph_result                  │                                │
       }                                           └────────────────────────────────┘
```

### 6.2 流程记忆路径

```
外部调用 Memory.add(messages, memory_type="process_memory", ...)
    │
    ▼
ProcessMemoryAddEngine.add(summaries, ...)
    │
    ├─ preprocess       : summaries → goals[], steps[], deps[], entity_type_map, task_description
    │                                                            ┌──────────────────┐
    ├─ search           : ProcessMemorySearchEngine             │ ← "search"      │
    │                    .search_for_dedup(goals, task_desc)     │   三层去重召回   │
    │                    → recalled = {graph{chains}, chunks, summaries}             │
    │                                                            └──────────────────┘
    ├─ decide           : LLM(summaries + recalled)
    │                    → {graph.steps[], graph.edges[], chunks[], summary}
    │
    ├─ execute          : 三层独立写入
    │   ├─ Graph        : graph.ingest(relations, node_properties={brief,goal,action,brief_embedding})
    │   ├─ Chunk        : vector_store.insert/update(memory_type="process_chunk")
    │   └─ Summary      : vector_store.insert/update(memory_type="process_summary")
    │                                               ┌────────────────────────────────┐
    └─ assemble         : final = {               │ ← "back"                        │
         "results": {graph_result, chunks, summary},│   三层写入结果 + 三层去重召回   │
         "recalled": recalled                       │   一起返回给调用方             │
       }                                           └────────────────────────────────┘
```

### 6.3 Flow 2 检索（独立于 Add）

```
外部 Agent 执行某一步时
    │
    ▼
Memory.search_process(current_step, previous_step, ...)
    │
    ▼
ProcessMemorySearchEngine.search_for_step()
    │
    ├─ Graph    : Brief embedding → search_nodes_by_embedding → 1-hop expand → semantic filter
    ├─ Chunk    : Goal → vector_store.search(memory_type="process_chunk")
    └─ Summary  : task_estimate → vector_store.search(memory_type="process_summary")
    │
    → {graph: {matched, expanded, filtered}, chunks: [...], summaries: [...]}
```

---

## 七、Vector Store 与 Neo4j Schema

### 7.1 标准记忆向量 payload

```python
{
    "data": "Name is John",           # embedding 文本
    "hash": "md5",
    "user_id": "u1",
    "agent_id": "a1",                 # 可选
    "run_id": "r1",                   # 可选
    "actor_id": "John",               # 可选
    "role": "user",                   # 可选
    "created_at": "2024-01-15T10:30:00+00:00",
    "updated_at": "..."
}
```

### 7.2 `process_chunk` payload

```python
{
    "memory_type": "process_chunk",
    "goal": "Add user authentication",
    "steps": [
        {"step": "01 - Read main.py", "brief": "Read main.py to understand entry point"},
        {"step": "03 - Create auth.py", "brief": "Create auth.py and implement login/logout"}
    ],
    "data": "Add user authentication",    # embedding 文本
    "hash": "md5",
    "user_id": "u1",
    "agent_id": "a1",                     # 可选
    "run_id": "r1",                       # 可选
    "created_at": "...",
    "updated_at": "..."
}
```

### 7.3 `process_summary` payload

```python
{
    "memory_type": "process_summary",
    "task_description": "Implement complete user authentication system",
    "full_chain": [
        {"step": "01 - Read main.py", "brief": "..."},
        {"step": "02 - Create config.py", "brief": "..."},
        {"step": "03 - Create auth.py", "brief": "..."}
    ],
    "data": "Implement complete user authentication system",
    "hash": "md5",
    "user_id": "u1",
    "created_at": "...",
    "updated_at": "..."
}
```

### 7.4 Neo4j Step 节点

```
(:Step {
    name: "03 - Create auth.py",
    user_id: "u1",
    agent_id: "a1",             # 可选
    run_id: "r1",               # 可选
    brief: "Create auth.py and implement login/logout functions",
    goal: "Add user authentication",
    action: "create_file(path='auth.py')",
    brief_embedding: [0.012, -0.034, ...],    # Vector property (via db.create.setNodeVectorProperty)
    mentions: 3,
    created: <timestamp>
})
```

### 7.5 Neo4j 标准 Entity 节点

```
(:Entity|:`__Entity__` {
    name: "john",
    user_id: "u1",
    embedding: [0.012, -0.034, ...],
    mentions: 3,
    created: <timestamp>
})
```

---

## 八、LangGraph 图映射

| Engine | 节点数 | 条件分支 | LLM 总调用 | LangGraph 变量 |
|--------|--------|---------|-----------|---------------|
| `SearchEngine` | 6 | 2 (`_should_search_graph`, `_should_rerank`) | 0-1（仅图搜索提取实体） | `SearchState` |
| `AddEngine` | 8 | 2 (`_should_infer`, `_should_extract_graph`) | 3（extract_queries + decide_memory + extract_graph） | `AddState` |
| `ProcessMemoryAddEngine` | 5 | 0（全路径） | 1（decide） | `ProcessAddState` |
| `ProcessMemorySearchEngine` | — | — | 0 | —（无 LangGraph） |

---

## 九、调用关系图

```
main.py
│
├── Memory.search(query, ...)                            ✅ 已委托
│     └── SearchEngine.search(query, filters, ...)
│           ├── embed        : embedding_model.embed()
│           ├── vector_search: vector_store.search()
│           ├── graph_search : graph.search_nodes()
│           ├── merge        : 按 id 去重
│           ├── rerank       : reranker.rerank() (可选)
│           └── build_response
│
├── Memory.add(messages, ..., memory_type=None)          ✅ 已委托
│     └── AddEngine.add(messages, metadata, filters, infer)
│           ├── preprocess    : messages → parsed_messages
│           ├── extract_queries: LLM → search_queries[]
│           ├── search        : SearchEngine.search() ← 加粗：add-with-search-back 的 "search"
│           ├── decide_memory : LLM → ADD/UPDATE/DELETE/NONE
│           ├── execute_vector: vector_store.insert/update/delete
│           ├── extract_graph : LLM → entity_type_map, relations, to_be_deleted
│           ├── execute_graph : graph.ingest()
│           └── assemble_result: results + recalled_memories + relations ← 加粗：add-with-search-back 的 "back"
│
├── Memory.add(messages, ..., memory_type="process_memory")  ✅ 已委托
│     └── ProcessMemoryAddEngine.add(summaries, metadata, filters)
│           ├── preprocess : summaries → goals/steps/deps/entity_type_map/task_desc
│           ├── search     : ProcessMemorySearchEngine.search_for_dedup()  ← "search"
│           ├── decide     : LLM → 三层 ADD/UPDATE/MERGE/NONE
│           ├── execute    : Graph(ingest) + Chunk(vector) + Summary(vector)
│           └── assemble   : results + recalled ← "back"
│
└── Memory.search_process(current_step, ...)             ✅ 已委托
      └── ProcessMemorySearchEngine.search_for_step()
            ├── Graph : search_nodes_by_embedding → expand_neighbors → semantic_filter
            ├── Chunk : vector_store.search(memory_type="process_chunk")
            └── Summary: vector_store.search(memory_type="process_summary")
```

---

## 十、错误处理与降级

| 场景 | 行为 |
|------|------|
| `memory_type="process_memory"` 但 `config.process_memory=None` | `Mem0ValidationError` (VALIDATION_005) |
| `search_process()` 但 `config.process_memory=None` | `Mem0ValidationError` (VALIDATION_005) |
| Search 图 db 不可用 | `graph_results=[]`, 仅返回向量结果 |
| Graph store 不可用时 | `enable_graph=False`，图节点被跳过 |
| LLM decide 返回非法 JSON | 返回空 `{}`，不抛异常（Add 结果为空） |
| UPDATE/MERGE 引用的 id 不在 recalled 中 | 降级为 ADD（id=null） |
| Search 引擎 `search_for_dedup` 异常 | 捕获后返回 `{"graph": {"chains": []}, "chunks": [], "summaries": []}` |
| 向量库搜索异常 | 返回 `[]`，记录 error 日志 |

---

## 十一、关键设计决策

1. **Search 是 Add 的依赖，不是反过来**：Add Engine 内部调用 Search Engine，Search Engine 不感知 Add
2. **"add-with-search-back"**：Add 返回结果中必须包含 `recalled_memories`（标准路径）或 `recalled`（流程路径），让外部 Agent 感知记忆联想
3. **流程记忆融入统一框架**：流程记忆的 `memory_type` 只影响数据解析方式，不影响 "召回→决策→执行→组装" 的整体流程
4. **三层写入完全独立**：ProcessMemoryAddEngine 的 Graph/Chunk/Summary 无写入先后依赖
5. **Flow 1 / Flow 2 读写分离**：半成品不污染记忆
6. **图存储复用 `MemoryGraph`**：通过 `node_label`（Entity vs Step）区分两种图模型，`ingest()` 通过 `node_properties` 参数区分标准记忆和流程记忆
7. **语义匹配在 Engine 层完成**：`search_nodes_by_embedding` 只做 Cypher 查询，embedding 由 ProcessMemorySearchEngine 外部计算
8. **LangGraph 状态机收在 Engine 内部**：不单独拆出 langgraph 文件，对外只暴露简单的 `search()` / `add()` 方法
9. **前一步语义筛选是 cosine 相似度**：不是 ID 精确匹配，而是用 Brief embedding 的余弦相似度筛选相关节点
