# Search Engine 重构实施计划

## 上下文

将 `mem0/memory/main.py` 中的 search 逻辑提取到独立的 `search_engine.py`，使用 LangGraph 编排召回流程（embed → vector_search → graph_search → merge_dedup → rerank），使 `main.py` 仅保留接口层和委托调用。

当前 search 是命令式代码（sync/async 各一套），直接调用 vector_store.search、graph.search、reranker.rerank，没有统一的召回编排层。重构后 Search Engine 成为 Add Engine 的依赖，Add 调用 Search 获取已有记忆后再做写入决策。

**本次重构原则**：
- 全部用 LangGraph 编排，不存在命令式回退路径
- 所有参数通过方法签名直接传入，不新增全局配置字段
- 本次只重构 Search，不碰 AsyncMemory

---

## 新增/修改文件

| 文件 | 操作 |
|------|------|
| `mem0/memory/search_engine.py` | 新增：SearchEngine 类 + LangGraph 状态机 |
| `mem0/memory/main.py` | 修改：初始化 SearchEngine，search() 方法委托调用 |
| `tests/memory/test_search_engine.py` | 新增：SearchEngine 专项测试 |

---

## 实施步骤（每个步骤可独立编写测试验证）

### Phase 0: 环境准备

项目使用 **conda `mem0` 环境**（Python 3.11.15），langgraph 1.1.6 已预装，无需额外安装。

#### 步骤 0.1: 确认 conda 环境激活
- 所有命令在 `mem0` conda 环境下执行：`conda activate mem0`
- **验证**: `which python` 应指向 `/home/wowoow/miniconda3/envs/mem0/bin/python`

#### 步骤 0.2: 启动 Docker 依赖服务
- docker-compose 文件位置：`server/docker-compose.yaml`
- 启动命令：
  ```bash
  cd server && docker compose up -d
  ```
- 服务包括：
  - **postgres** (pgvector): 端口 `8432` → 容器 `5432`
  - **neo4j**: Bolt 端口 `8687`，HTTP 端口 `8474`
- **验证**: `docker compose ps` 两个服务状态均为 `healthy`

#### 步骤 0.3: 加载 API Key 环境变量
- `.env` 文件位置：`server/.env`
- 加载方式：`export $(grep -v '^#' server/.env | xargs)` 或在测试代码中用 `python-dotenv` 加载
- 关键变量：
  - `OPENAI_llm_API_KEY` / `OPENAI_llm_URL` / `OPENAI_llm_Model`
  - `OPENAI_EMBEDDER_API_KEY` / `OPENAI_EMBEDDER_URL` / `OPENAI_EMBEDDER_MODEL`
  - `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`
  - `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_USER` / `POSTGRES_PASSWORD`
- **验证**: `echo $OPENAI_llm_API_KEY` 输出非空

---

### Phase 1: 基础设施准备

#### 步骤 1.1: 确认 `_build_filters_and_metadata` 可 import 复用
- 该函数当前在 `main.py` 第159行，是 search 和 add 的公共依赖
- 保持不变，确认 `search_engine.py` 可以直接 `from mem0.memory.main import _build_filters_and_metadata`
- **测试验证**: 在 `tests/memory/test_search_engine.py` 中编写 `test_import_build_filters`，验证 import 成功

---

### Phase 2: SearchEngine 核心节点方法

#### 步骤 2.1: 创建 `SearchEngine` 类骨架和 `__init__`
- 文件：`mem0/memory/search_engine.py`
- 构造函数接收：`embedding_model`, `vector_store`, `graph_store` (可选), `reranker` (可选)
- 存储各组件引用，提供 `enable_graph` 属性（由 graph_store 是否非空决定）
- **测试验证**: 编写 `test_search_engine_init` — 分别测试有 graph/无 graph、有 reranker/无 reranker 的初始化

#### 步骤 2.2: 实现 `_embed_query` 节点方法
- 调用 `self.embedding_model.embed(query, "search")` 生成 embedding
- 返回 embedding vector
- **测试验证**: 编写 `test_embed_query` — mock embedding_model，验证调用参数和返回值

#### 步骤 2.3: 实现 `_search_vector_store` 节点方法
- 接收 query, embeddings, filters, limit, threshold
- 调用 `self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)`
- 将返回结果转换为 `MemoryItem` dict 列表（复用现有 `main.py` 的转换逻辑，提取为 `_format_vector_results` 静态方法）
- 应用 threshold 过滤
- **测试验证**: 编写 `test_search_vector_store_basic` 和 `test_search_vector_store_threshold_filtering`
  - mock vector_store 返回带 payload 的 Memory 对象
  - 验证返回列表结构正确（含 id, memory, hash, score, created_at, updated_at, metadata 等）
  - 验证 threshold 低于/高于 score 时的包含/排除行为

#### 步骤 2.4: 实现 `_search_graph` 节点方法
- 接收 query, filters, graph_depth
- 若 `graph_depth <= 0` 或 `not self.enable_graph`，直接返回空列表
- 否则调用 `self.graph.search(query, filters, limit=graph_depth)`
- **注意**：graph.search 的 limit 参数语义是"遍历深度"，不是结果数量
- **测试验证**: 编写 `test_search_graph_disabled`、`test_search_graph_depth_zero`、`test_search_graph_with_depth`
  - mock graph.search 返回关系列表
  - 验证 graph_depth=0 时不会调用 graph.search
  - 验证 enable_graph=False 时不会调用 graph.search

#### 步骤 2.5: 实现 `_merge_results` 节点方法（合并与去重）
- 接收 vector_results（MemoryItem 列表）和 graph_results（关系列表）
- 按 memory id 对 vector_results 去重（保留 score 最高的）
- 返回合并后的结果结构：`{"vector_results": [...], "graph_results": [...]}`
- **测试验证**: 编写 `test_merge_results_dedup_by_id` — 构造有重复 id 的 vector_results，验证只保留一个
  - 编写 `test_merge_results_empty_inputs` — 验证空列表不会报错

#### 步骤 2.6: 实现 `_rerank_results` 节点方法
- 接收 query, vector_results, limit
- 若 `rerank=False` 或 `self.reranker is None`，直接返回原结果
- 否则调用 `self.reranker.rerank(query, vector_results, limit)`
- **测试验证**: 编写 `test_rerank_disabled`、`test_rerank_no_reranker`、`test_rerank_success`
  - mock reranker，验证调用参数正确
  - 验证 rerank=False 时不调用 reranker

#### 步骤 2.7: 实现 `_build_search_response` 节点方法
- 接收 merged_results, enable_graph
- 组装最终返回结构：
  - 若 enable_graph: `{"results": vector_results, "relations": graph_results}`
  - 否则: `{"results": vector_results}`
- **测试验证**: 编写 `test_build_response_with_graph` 和 `test_build_response_without_graph`

---

### Phase 3: LangGraph 状态机编排

#### 步骤 3.1: 定义 `SearchState` TypedDict
- 在 `search_engine.py` 中定义状态 schema：
```python
class SearchState(TypedDict):
    query: str
    filters: dict
    limit: int
    threshold: Optional[float]
    graph_depth: int
    rerank: bool
    embedding: Optional[list]          # _embed_query 输出
    vector_results: list               # _search_vector_store 输出
    graph_results: list                # _search_graph 输出
    merged_results: dict               # _merge_results 输出
    final_results: dict                # _build_search_response 输出
    error: Optional[str]               # 错误信息
```
- **测试验证**: 编写 `test_search_state_schema` — 验证各字段类型约束

#### 步骤 3.2: 包装节点函数适配 LangGraph
- 将 2.2-2.7 的方法包装为接收 `SearchState` 返回 `dict` 的节点函数
- 每个节点只读取自己需要的 state 字段，返回要更新的字段 dict
- **测试验证**: 对每个节点函数编写独立测试，传入完整 SearchState dict，验证返回的更新 dict 正确

#### 步骤 3.3: 实现条件边逻辑
- `should_search_graph(state) -> Literal["graph_search", "merge"]`：当 graph_depth > 0 且 enable_graph 为 True 时返回 "graph_search"，否则返回 "merge"
- `should_rerank(state) -> Literal["rerank", "build_response"]`：当 rerank=True 且 reranker 存在且 vector_results 非空时返回 "rerank"，否则返回 "build_response"
- **测试验证**: 编写 `test_should_search_graph_conditions` 和 `test_should_rerank_conditions`
  - 覆盖所有分支组合（graph_depth > 0 / <= 0, enable_graph True/False, rerank True/False 等）

#### 步骤 3.4: 编译 LangGraph 状态机
- 在 `SearchEngine.__init__` 中：
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(SearchState)
builder.add_node("embed", self._node_embed)
builder.add_node("vector_search", self._node_vector_search)
builder.add_node("graph_search", self._node_graph_search)
builder.add_node("merge", self._node_merge)
builder.add_node("rerank", self._node_rerank)
builder.add_node("build_response", self._node_build_response)

builder.add_edge(START, "embed")
builder.add_edge("embed", "vector_search")
builder.add_conditional_edges("vector_search", self._should_search_graph, {
    "graph_search": "graph_search",
    "merge": "merge",
})
builder.add_edge("graph_search", "merge")
builder.add_edge("merge", "rerank")
builder.add_conditional_edges("rerank", self._should_rerank, {
    "rerank": "rerank",
    "build_response": "build_response",
})
builder.add_edge("rerank", "build_response")
builder.add_edge("build_response", END)

self.search_graph = builder.compile()
```
- **测试验证**: 编写 `test_search_graph_compilation` — 验证图编译成功，无循环错误

#### 步骤 3.5: 实现 `SearchEngine.search()` 入口方法
- 接收参数：`query`, `filters`, `limit=100`, `threshold=None`, `graph_depth=2`, `rerank=True`
- 构造初始 `SearchState`
- 调用 `self.search_graph.invoke(state)`
- 从结果中提取 `final_results` 返回
- **测试验证**: 编写 `test_search_entry_full_pipeline` — mock 所有依赖组件，验证完整 LangGraph 链路正确返回结果结构

---

### Phase 4: main.py 集成

#### 步骤 4.1: 在 `Memory.__init__` 中初始化 SearchEngine
- 在 `main.py` `Memory.__init__`（约第245行）末尾添加：
```python
from mem0.memory.search_engine import SearchEngine
self.search_engine = SearchEngine(
    embedding_model=self.embedding_model,
    vector_store=self.vector_store,
    graph_store=self.graph if self.enable_graph else None,
    reranker=self.reranker,
)
```
- **测试验证**: 编写 `test_memory_init_creates_search_engine` — mock SearchEngine 类，验证构造函数被正确调用

#### 步骤 4.2: 重写 `Memory.search()` 为委托调用
- 保留 `search()` 的完整方法签名和参数校验逻辑（lines 840-898 的 filters、telemetry 等）
- 将实际的召回逻辑替换为：
```python
return self.search_engine.search(
    query=query,
    filters=effective_filters,
    limit=limit,
    threshold=threshold,
    graph_depth=2,   # 默认图跨步深度，通过参数传入
    rerank=rerank,
)
```
- **测试验证**: 编写 `test_memory_search_delegates_to_engine` — mock search_engine.search，验证参数传递正确

#### 步骤 4.3: 删除 `main.py` 中原 `_search_vector_store` 方法
- 同步版本（lines 1036-1072）和异步版本（lines 2106-2144）
- **测试验证**: 运行 `pytest tests/test_main.py tests/test_memory.py` 确认无引用断裂

---

### Phase 5: 端到端验证

#### 步骤 5.1: 运行现有测试套件
```bash
pytest tests/test_main.py tests/test_memory.py tests/memory/test_main.py -v
```
- 所有现有测试必须通过，确保重构无回归

#### 步骤 5.2: 编写 SearchEngine Mock 集成测试
- `test_search_engine_end_to_end_vector_only`：仅向量召回，无 graph，无 rerank
- `test_search_engine_end_to_end_with_graph`：向量 + graph，graph_depth=2
- `test_search_engine_end_to_end_with_rerank`：向量 + rerank
- `test_search_engine_end_to_end_full_pipeline`：向量 + graph + rerank 完整链路
- 使用 unittest.mock 模拟 embedding_model、vector_store、graph、reranker

#### 步骤 5.3: 编写 SearchEngine 真实 Factory E2E 测试
- **环境要求**：docker 服务已启动（postgres + neo4j），`server/.env` 已加载
- 通过真实 Factory 初始化底层组件（不经过 Memory 上层）：
  - `embedding_model = EmbedderFactory.create("openai", ...)` → 使用 .env 中的 embedder API key
  - `vector_store = VectorStoreFactory.create("pgvector", ...)` → 连接本地 postgres:8432
  - `graph_store = GraphStoreFactory.create("default", ...)` → 连接本地 neo4j:8687
  - 用上述组件直接构造 `SearchEngine`
- 测试数据准备：直接调用底层 API，不经过 `Memory.add()`：
  - 向量数据：`vector_store.insert(vectors=[...], payloads=[{"data": "test memory"}], ids=["test-id"])`
  - 图数据：`graph_store.add(data="entity1 relation entity2", filters={"user_id": "test"})`
- 测试前清理：`vector_store.reset()` / `graph_store.delete_all()`
- E2E 测试用例：
  - `test_e2e_search_vector_recall`：直接 insert 向量后 search 能召回
  - `test_e2e_search_graph_recall`：直接写入图数据后通过 graph_depth > 0 召回关系
  - `test_e2e_search_with_rerank`：配置真实 reranker 后验证排序变化
- **测试验证**: 每条测试真实写入、真实查询、真实返回，验证结果结构和内容正确

#### 步骤 5.4: 验证 LangGraph 图结构正确性
- 使用 `search_graph.get_graph()` 获取图结构
- 验证节点和边的存在性
- **测试验证**: 编写 `test_langgraph_structure`，断言节点列表包含 ["embed", "vector_search", "graph_search", "merge", "rerank", "build_response"]

#### 步骤 5.5: 手动验证 main.py 的 search 接口行为一致
- 使用 mock fixture（如 `memory_instance`）调用 `search()`
- 验证返回结构与重构前一致：`{"results": [...], "relations": [...]}`（v1.1+ 且 enable_graph=True 时）

---

## 关键复用点

| 功能 | 来源 | 复用方式 |
|------|------|----------|
| filters/metadata 构建 | `main.py:159` `_build_filters_and_metadata` | import 直接使用 |
| 向量结果格式化 | `main.py:1036` `_search_vector_store` 中的 MemoryItem 转换逻辑 | 提取为静态方法复用 |
| timestamp 归一化 | `main.py:53` `_normalize_iso_timestamp_to_utc` | import 直接使用 |
| telemetry filters | `main.py:35` `process_telemetry_filters` | 保留在 main.py 中，search 前调用 |
| advanced filter 处理 | `main.py:940` `_process_metadata_filters` | 保留在 main.py 中，search 前调用 |

---

## 接口契约

### SearchEngine.search() 签名
```python
def search(
    self,
    query: str,
    filters: dict,
    limit: int = 100,
    threshold: Optional[float] = None,
    graph_depth: int = 2,
    rerank: bool = True,
) -> dict:
```

### 返回值
```python
# 有图存储:
{"results": [MemoryItem, ...], "relations": [{"source": str, "relationship": str, "destination": str}, ...]}

# 无图存储:
{"results": [MemoryItem, ...]}
```

### Memory.search() 对外接口不变
- 方法签名、参数、返回值结构与重构前完全一致
- 调用方无感知

---

## 风险与回退

1. **Graph search 语义变化**：当前 graph.search 的 limit 参数语义是"深度"，重构后显式使用 `graph_depth` 参数传递，避免混淆
2. **AsyncMemory 不修改**：本次只重构 Memory.search()，AsyncMemory 保持原样，后续可参照同步版本改造
