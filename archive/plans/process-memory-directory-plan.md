# Process Memory 目录结构

## 一、现有架构（不动）

```
mem0/memory/
├── main.py                  # Memory + AsyncMemory：接口层，初始化各 Engine，委托调用
├── base.py                  # MemoryBase（不动）
├── storage.py               # SQLiteManager：history 记录（不动）
├── telemetry.py             # 遥测（不动）
├── utils.py                 # 通用工具（不动）
├── graph_memory.py          # MemoryGraph：标准记忆图存储（复用其 ingest/search_nodes 等非 LLM 接口）
├── memgraph_memory.py       # 已有
├── kuzu_memory.py           # 已有
├── apache_age_memory.py     # 已有
├── setup.py                 # 已有
├── search_engine.py         # 标准记忆 SearchEngine
└── add_engine.py            # 标准记忆 AddEngine
```

---

## 二、新增文件

```
mem0/memory/
├── ...（以上不动）
│
├── process_search_engine.py   # 【新增】Process Memory Search Engine
└── process_add_engine.py      # 【新增】Process Memory Add Engine
```

不新增 `process_graph_store.py`。图操作直接复用 `MemoryGraph` 的 `ingest()`、`search_nodes()`、`_search_graph_db_by_depth()` 等非 LLM 接口，通过不同的 `node_label` 区分 Step 节点和 Entity 节点。

---

## 三、文件职责与接口

### process_search_engine.py

Process Memory 统一检索引擎。Flow 1（去重搜索）和 Flow 2（过程搜索）共用，内部分别调用三层搜索。

```python
class ProcessMemorySearchEngine:
    def __init__(
        self,
        embedding_model,          # EmbedderFactory 创建的 embedding 实例
        vector_store,             # Chunk + Summary 的向量存储
        graph_store,              # MemoryGraph 实例（配置为 Step node_label）
    ): ...

    # —— Flow 2：过程检索 ——
    def search_for_step(
        self,
        current_step: dict,           # 当前 step 的 summary
        previous_step: dict | None,   # 前一步的 summary，用于语义再筛选（不是 ID 匹配）
        filters: dict,                # user_id/agent_id/run_id
        task_estimate: str | None = None,  # 任务类型估计，None 时用 current_step["Goal"] 拼接
        graph_hop: int = 1,
        chunk_top_k: int = 5,
        summary_top_k: int = 3,
    ) -> dict:
        """
        Flow 2 返回:
        {
            "graph": {
                "matched_nodes": [...],    # _search_graph_by_brief 匹配的节点
                "expanded_nodes": [...],   # 1-hop 扩展邻居
                "filtered_nodes": [...],   # 前一步语义筛选后
            },
            "chunks": [
                {"goal": str, "score": float, "steps": [...], "id": str|None, "metadata": dict}
            ],
            "summaries": [
                {"task_description": str, "score": float, "full_chain": [...], "id": str|None, "metadata": dict}
            ],
        }
        """

    # —— Flow 1：去重检索 ——
    def search_for_dedup(
        self,
        goals: list[str],                  # 从 summaries 提取的去重 Goal 列表
        task_description: str | None,      # 任务宏观描述
        filters: dict,
    ) -> dict:
        """
        Flow 1 返回:
        {
            "graph": {
                "chains": [...]            # _search_graph_by_step_names 遍历的完整链路
            },
            "chunks": [...],               # 同 Flow 2
            "summaries": [...],            # 同 Flow 2
        }
        """

    # —— 内部私有 ——
    def _embed(self, text: str) -> list[float]: ...

    # 图搜索（复用 MemoryGraph 非 LLM 接口）
    def _search_graph_by_brief(
        self, brief: str, filters: dict, top_k: int
    ) -> list[dict]:
        """Brief embedding → MemoryGraph.search_nodes_by_embedding → 返回匹配节点列表"""

    def _expand_neighbors(
        self, node_names: list[str], filters: dict, depth: int
    ) -> list[dict]:
        """调用 graph_store.search_nodes(node_names, filters, depth=depth)"""

    def _semantic_filter(
        self, nodes: list[dict], previous_step: dict, threshold: float
    ) -> list[dict]:
        """用前一步 Brief embedding 与候选节点 Brief 做余弦相似度筛选"""

    def _search_graph_by_step_names(
        self, step_names: list[str], filters: dict, depth: int = 10
    ) -> list[dict]:
        """Chunk 返回的 step 名称 → MemoryGraph.search_nodes 遍历完整链路"""

    # 向量搜索（直接调 vector_store）
    def _search_chunks(
        self, goal: str, filters: dict, top_k: int
    ) -> list[dict]: ...

    def _search_summaries(
        self, task_desc: str, filters: dict, top_k: int
    ) -> list[dict]: ...
```

---

### process_add_engine.py

Process Memory 写入引擎。仅 Flow 1 使用。内部通过 LangGraph 编排（preprocess → search_for_dedup → LLM decide → execute → assemble）。

```python
class ProcessMemoryAddEngine:
    def __init__(
        self,
        embedding_model,
        vector_store,
        llm,
        db,                  # SQLiteManager
        search_engine,       # ProcessMemorySearchEngine
        graph_store,         # MemoryGraph 实例（配置为 Step node_label）
    ): ...

    def add(
        self,
        summaries: list[dict],      # 完整 summary 数组（按 step 顺序）
        metadata: dict,             # 含 user_id/agent_id/run_id
    ) -> ProcessAddResult:
        """
        内部 LangGraph 编排:
        1. preprocess: 解析 summaries → steps / deps / goals / task_description
        2. search: self.search_engine.search_for_dedup(goals, task_desc, filters)
        3. decide: LLM(新 summaries + 三层 recall) → 三层 ADD/UPDATE/MERGE/NONE
        4. execute: 三层独立并行写入
        5. assemble: 组装返回
        """
```

返回结构：

```python
# add() 返回 dict，结构为 {"results": {...}, "recalled": {...}}
```

LLM 决策输出格式（`_decide_process_memory` prompt 产出的 JSON）：

```json
{
  "graph": {
    "steps": [
      {"name": "03 - 创建 auth.py", "event": "ADD",
       "goal": "添加用户认证功能", "brief": "...", "action": "create_file(path='auth.py')"}
    ],
    "edges": [
      {"source": "01 - 阅读 main.py", "target": "03 - 创建 auth.py",
       "relationship": "DEPENDS_ON", "event": "ADD"}
    ]
  },
  "chunks": [
    {"goal": "添加用户认证功能", "event": "MERGE", "merge_with": "<existing_id>",
     "steps": [...]}
  ],
  "summary": {
    "event": "ADD",
    "task_description": "...",
    "full_chain": [{"step": "...", "brief": "..."}]
  }
}
```

---

## 四、配置扩展

`mem0/configs/base.py` — `MemoryConfig` 新增：

```python
process_memory: Optional[ProcessMemoryConfig] = None
```

```python
class ProcessMemoryConfig(BaseModel):
    vector_store: VectorStoreConfig     # Chunk + Summary 向量存储
    graph_store: GraphStoreConfig       # Step 依赖图的 Neo4j 配置
    graph_search_depth: int = 1         # Flow 2 图扩展深度（默认 1-hop）
    chunk_top_k: int = 5
    summary_top_k: int = 3
    semantic_filter_threshold: float = 0.6  # 前一步语义筛选阈值
```

---

## 五、main.py 变更

```python
class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig):
        # ... 已有初始化 ...
        
        # 标准记忆引擎（已有）
        self.search_engine = SearchEngine(...)
        self.add_engine = AddEngine(...)
        
        # Process Memory 引擎（新增，可选）
        if config.process_memory is not None:
            self.process_graph_store = MemoryGraph(config)  # 复用 MemoryGraph，node_label 区分
            self.process_search_engine = ProcessMemorySearchEngine(
                embedding_model=self.embedding_model,
                vector_store=_init_vector_store(config.process_memory.vector_store),
                graph_store=self.process_graph_store,
            )
            self.process_add_engine = ProcessMemoryAddEngine(
                embedding_model=self.embedding_model,
                vector_store=...,
                llm=self.llm,
                db=self.db,
                search_engine=self.process_search_engine,
                graph_store=self.process_graph_store,
            )

    def add(self, messages, ..., memory_type=None):
        if memory_type == "procedural_memory":
            return self.process_add_engine.add(...)
        # 其余走已有 AddEngine

    def search_process(self, current_step, previous_step, filters, task_estimate=None):
        """Flow 2 入口：外部 Agent 过程检索。不在现有 search() 里，签名完全不同"""
        return self.process_search_engine.search_for_step(
            current_step, previous_step, filters, task_estimate=task_estimate
        )
```

---

## 六、测试文件

```
tests/memory/
├── test_search_engine.py                # 已有
├── test_add_engine.py                   # 已有
├── test_add_engine_e2e.py               # 已有
├── test_process_search_engine.py        # 【新增】Process Search 单元测试
├── test_process_add_engine.py           # 【新增】Process Add 单元测试
└── test_process_e2e.py                  # 【新增】Process Memory E2E 测试
```

---

## 七、调用关系图

```
main.py
│
├── add(memory_type="procedural_memory")                    ← Flow 1 入口
│     └── ProcessMemoryAddEngine.add()
│           │
│           ├── preprocess : 解析 summaries → steps/deps/goals/task_desc
│           │
│           ├── search    : 调用 ProcessMemorySearchEngine.search_for_dedup(goals, task_desc, filters)
│           │     │
│           │     ├── Chunk    : _search_chunks(goal)                                      ← Step 1
│           │     │                └── vector_store.search(goal_text, filters={memory_type: "process_chunk"})
│           │     │
│           │     ├── Graph    : 从 Chunk 结果取已有 step 名称                                ← Step 2（依赖 Chunk）
│           │     │                └── MemoryGraph.search_nodes(step_names, depth=10) 沿 DEPENDS_ON 全链路遍历
│           │     │
│           │     └── Summary  : _search_summaries(task_desc)                              ← 独立
│           │                      └── vector_store.search(task_desc, filters={memory_type: "process_summary"})
│           │
│           ├── decide    : LLM(新 summaries + 三层 recall) → 决策 JSON
│           │
│           ├── execute   : 三层独立写入（无先后依赖）
│           │     ├── Graph    : MemoryGraph.ingest(entity_type_map, relations, filters, node_properties)
│           │     ├── Chunk    : vector_store.insert/update()
│           │     └── Summary  : vector_store.insert/update()
│           │
│           └── assemble  : ProcessAddResult(results + recalled)
│
│
├── search_process(current_step, previous_step, filters, task_estimate=None)  ← Flow 2 入口（新增）
│     └── ProcessMemorySearchEngine.search_for_step()
│           │
│           ├── Graph    : _search_graph_by_brief(current_step["Brief"])
│           │                └── MemoryGraph.search_nodes_by_embedding(embedding, filters)  ← semantic match
│           │             → _expand_neighbors(matched_node_names, depth=1)
│           │                └── MemoryGraph.search_nodes(node_names, filters)              ← exact traversal
│           │             → _semantic_filter(expanded_nodes, previous_step)                 ← cosine filter
│           │
│           ├── Chunk    : _search_chunks(current_step["Goal"])
│           │                └── vector_store.search(goal_text, filters={memory_type: "process_chunk"})
│           │
│           └── Summary  : _search_summaries(task_estimate or _build_task_estimate(current_step))
│                            └── vector_store.search(task_text, filters={memory_type: "process_summary"})
│
│
├── search(query, ...)                                      ← 已有，不动
│     └── SearchEngine.search()
│
└── add(messages, ...)                                      ← 已有，不动
      └── AddEngine.add()
```

---

## 八、关键设计决策

1. **不新建 ProcessGraphStore**：直接复用 `MemoryGraph` 的非 LLM 接口（`ingest`/`search_nodes`/`search`/`_search_graph_db_by_depth`），通过 `node_label` 区分 Step 节点和 Entity 节点
2. **语义匹配在 SearchEngine 层完成**：Brief embedding 相似度匹配在 ProcessMemorySearchEngine 内部用 embedding_model 完成，MemoryGraph 只做精确遍历
3. **前一步再筛选是语义的**：用前一步 Brief 的 embedding 与候选节点做余弦相似度筛选，不是 ID 精确匹配
4. **LLM 统一决策三层**：和 AddEngine.decide_memory 模式一致，把 召回结果 + 新输入 一起给 LLM 决策
5. **三层写入并行独立**：Graph/Chunk/Summary 三者无写入先后依赖
6. **Flow 2 走独立入口**：不在现有 `search()` 里加参数，而是新增 `search_process()` 方法，签名和返回结构完全不同
7. **filters 在入口注入**：`user_id`/`agent_id`/`run_id` 初始化时确定，后续所有操作在同 ID 下进行
