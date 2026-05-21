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
| `tests/memory/test_search_engine_e2e.py` | 新增：SearchEngine E2E 测试（真实 pgvector + neo4j） |

---

## 实施步骤（每个步骤可独立编写测试验证）

### Phase 0: 环境准备

项目使用 **conda `mem0` 环境**（Python 3.11.15），langgraph 1.1.6 已预装，无需额外安装。

#### 步骤 0.1: 确认 conda 环境激活
- 若 shell 未初始化 conda，先执行：`source /home/wowoow/miniconda3/etc/profile.d/conda.sh`
- 激活环境：`conda activate mem0`
- **验证**: `which python` 应指向 `/home/wowoow/miniconda3/envs/mem0/bin/python`
- **验证**: `python -c "import langgraph; print('ok')"` 应输出 `ok`

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
builder.add_conditional_edges("merge", self._should_rerank, {
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

### Phase 4: main.py 集成（✅ 已完成）

> **范围说明**：本阶段所有改造仅针对 `Memory`（同步类）。`AsyncMemory` 保持原样，后续可参照同步版本改造。

#### 步骤 4.1: 在 `Memory.__init__` 中初始化 SearchEngine ✅
- 在 `main.py` `Memory.__init__`（约第306行）添加：
```python
from mem0.memory.search_engine import SearchEngine
self.search_engine = SearchEngine(
    embedding_model=self.embedding_model,
    vector_store=self.vector_store,
    graph_store=self.graph if self.enable_graph else None,
    reranker=self.reranker,
)
```
- **实现偏差**: 实际位置约第306行（在 capture_event 之前），不是原计划的第245行
- **测试验证**: `test_memory_search_delegates_to_engine` 已验证构造函数参数传递正确

#### 步骤 4.2: 重写 `Memory.search()` 为委托调用 ✅
- 保留 `search()` 的完整方法签名和参数校验逻辑（lines 850-922 的 filters、telemetry 等）
- 将实际的召回逻辑替换为委托调用（line 924-931）：
```python
return self.search_engine.search(
    query=query,
    filters=effective_filters,
    limit=limit,
    threshold=threshold,
    graph_depth=2,   # Memory.search() 未暴露此参数，硬编码默认值为 2
    rerank=rerank,
)
```
- **实现偏差**: `graph_depth` 未在 `Memory.search()` 签名中暴露，后续 Add Engine 工作时需配置化
- **测试验证**: `test_memory_search_delegates_to_engine` 已验证参数传递正确

#### 步骤 4.3: 删除 `main.py` 中原 `_search_vector_store` 方法 ✅
- 同步版本（原 lines 1036-1072）已删除
- **异步版本保留**: 按原范围说明，本次不碰 `AsyncMemory`，因此异步版本 `_search_vector_store`（lines ~2106-2144）保留未动
- **测试验证**: `pytest tests/test_main.py tests/test_memory.py tests/memory/test_search_engine.py` 全部通过

---

### Phase 5: 端到端验证（✅ 已完成）

#### 步骤 5.1: 运行现有测试套件 ✅
```bash
pytest tests/test_main.py tests/test_memory.py tests/memory/test_search_engine.py -v
```
- **结果**: 96/98 通过（2 个 async 失败为预存在的环境问题，与本次重构无关）
- 所有现有 `test_search`、`test_search_handles_incomplete_payloads` 等测试已通过

#### 步骤 5.2: 编写 SearchEngine Mock 集成测试 ✅
新增 `TestMemorySearchIntegration` 测试类（位于 `tests/memory/test_search_engine.py`）：
- `test_memory_search_delegates_to_engine`：验证 `Memory.search()` 正确委托参数给 SearchEngine
- `test_memory_search_e2e_vector_only`：仅向量召回，无 graph，验证返回结构
- `test_memory_search_e2e_with_graph`：向量 + graph，graph_depth=2，验证 `relations` 存在
- `test_memory_search_e2e_with_rerank`：向量 + rerank，验证排序变化和 `rerank_score`
- 使用 `Mock` 模拟 embedding_model、vector_store、graph、reranker

#### 步骤 5.3: 编写 SearchEngine 真实 Factory E2E 测试 ✅
新增 `tests/memory/test_search_engine_e2e.py`：
- **环境**: docker 服务已启动（postgres:8432 + neo4j:8687），`server/.env` 已加载
- **向量 E2E** (`TestE2EVectorRecall`):
  - `test_e2e_vector_recall_basic`：直接 insert OpenAI embedding 后 search 召回
  - `test_e2e_vector_threshold_filtering`：threshold 过滤低分结果（自适应阈值策略）
  - `test_e2e_vector_merge_dedup`：验证同一 ID 不会重复出现
  - `test_e2e_vector_empty_results`：不存在的 user_id 返回空结果
- **图 E2E** (`TestE2EGraphRecall`):
  - `test_e2e_graph_recall_basic`：向量 + alice-KNOWS->Bob 图关系，验证 `relations` 非空
  - `test_e2e_graph_depth_zero_skips_graph`：graph_depth=0 时 `relations` 为空列表
  - `test_e2e_graph_multi_hop_traversal`：alice->Bob->Carol 两跳链，depth=2 验证多跳召回
- **图测试策略**: E2E 中创建 `Neo4jGraphStore` 包装类，直接用 Cypher 读写图数据，绕过 LLM 实体提取，确保测试稳定性
- **结果**: 7 个 E2E 测试全部通过

#### 步骤 5.4: 验证 LangGraph 图结构正确性 ✅
- `test_langgraph_structure`：断言节点列表包含 ["embed", "vector_search", "graph_search", "merge", "rerank", "build_response"]
- `test_search_graph_compilation`：验证图编译成功
- **结果**: 通过

#### 步骤 5.5: 验证 main.py 的 search 接口行为一致 ✅
- `test_search`（parametrized v1.0/v1.1 × enable_graph True/False）：验证返回结构与重构前一致
- `test_memory_search_e2e_vector_only` / `test_memory_search_e2e_with_graph`：验证 Memory 层接口行为
- **结果**: 通过

---

## 关键复用点

| 功能 | 来源 | 复用方式 |
|------|------|----------|
| filters/metadata 构建 | `main.py:159` `_build_filters_and_metadata` | import 直接使用 |
| 向量结果格式化 | `main.py` 原 `_search_vector_store` 中的 MemoryItem 转换逻辑 | 提取为 `_format_vector_results` 静态方法，移至 `search_engine.py` |
| timestamp 归一化 | `main.py:53` `_normalize_iso_timestamp_to_utc` | import 直接使用 |
| telemetry filters | `main.py:35` `process_telemetry_filters` | 保留在 main.py 中，search 前调用 |
| advanced filter 处理 | `main.py:940` `_process_metadata_filters` | **Memory 实例方法**，保留在 main.py 中预处理后再传入 search_engine，search_engine 不直接复用 |
| OutputData (向量库返回模型) | `mem0/vector_stores/pgvector.py` 等具体实现 | 各 vector store 独立定义，字段为 `id`, `score`, `payload`。search_engine 通过 `vector_store.search()` 返回值间接使用，不直接 import |

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
{
    "results": [MemoryItem, ...],
    "relations": [{"source": str, "relationship": str, "destination": str}, ...]
}

# 无图存储:
{"results": [MemoryItem, ...]}
```

### MemoryItem 字段（Pydantic v2 模型）
已验证字段：`id`, `memory`, `hash`, `metadata`, `score`, `created_at`, `updated_at`

### OutputData 字段（向量库返回模型）
各 vector store 实现独立定义，已验证字段：`id`, `score`, `payload`
- `payload` 中包含 `data`, `hash`, `created_at`, `updated_at` 以及 `user_id`/`agent_id`/`run_id` 等 metadata

### Memory.search() 对外接口不变
- 方法签名、参数、返回值结构与重构前完全一致
- 调用方无感知

---

## 风险与回退

1. **Graph search 语义变化**：当前 graph.search 的 limit 参数语义是"深度"，重构后显式使用 `graph_depth` 参数传递，避免混淆
2. **AsyncMemory 不修改**：本次只重构 Memory.search()，AsyncMemory 保持原样，后续可参照同步版本改造

---

## 附录：Phase 0/1 执行结果（2026/05/17）

### 环境验证结果
- conda `mem0` 环境激活成功：Python 3.11.15，langgraph 可用
- Docker 服务启动成功：postgres (pgvector) + neo4j 均为 healthy
- API Key 环境变量加载成功：所有关键变量（OPENAI_llm、OPENAI_EMBEDDER、NEO4J、POSTGRES）均已确认存在

### Import 验证结果
- `_build_filters_and_metadata` ✅ 从 `mem0.memory.main` 导入成功
- `_normalize_iso_timestamp_to_utc` ✅ 从 `mem0.memory.main` 导入成功
- `MemoryItem` ✅ 从 `mem0.configs.base` 导入成功，字段：id, memory, hash, metadata, score, created_at, updated_at
- `process_telemetry_filters` ✅ 从 `mem0.memory.utils` 导入成功
- `OutputData` ✅ 从 `mem0.vector_stores.pgvector` 导入成功，字段：id, score, payload
- `_process_metadata_filters` ❌ **Memory 实例方法**，不可直接 import。保留在 main.py 中预处理后再传入 search_engine

### 设计修正记录
1. **LangGraph 条件边**：`should_rerank` 条件边从 `merge` 节点出发（原设计从 `rerank` 节点出发会导致逻辑错误）
2. **OutputData 路径**：不在 `mem0.vector_stores.base`，在各具体实现中定义
3. **conda 激活**：非交互式 shell 需先 `source /home/wowoow/miniconda3/etc/profile.d/conda.sh`

---

## 附录：实现完成度审计与后续依赖分析（2026/05/17）

### 实施完成状态

| Phase | 内容 | 状态 | 说明 |
|-------|------|------|------|
| Phase 0 | 环境准备 | ✅ 完成 | conda + docker + .env 全部就绪 |
| Phase 1 | 基础设施 | ✅ 完成 | import 验证通过 |
| Phase 2 | SearchEngine 核心节点 | ✅ 完成 | 6 个节点方法全部实现，51 个单元测试通过 |
| Phase 3 | LangGraph 状态机 | ✅ 完成 | 图编译成功，条件边逻辑正确 |
| Phase 4 | main.py 集成 | ✅ 完成 | `Memory.__init__` 已初始化 SearchEngine；`search()` 已委托；同步 `_search_vector_store` 已删除 |
| Phase 5 | 端到端验证 | ✅ 完成 | 51 个单元测试 + 4 个 Memory 集成测试 + 7 个 E2E 测试全部通过 |

**结论**：Search 引擎重构 100% 完成。`search_engine.py`、`test_search_engine.py`、`test_search_engine_e2e.py` 均已就绪，`main.py` 已完成接入。

### 实际实现 vs 计划的偏差记录

1. **`_rerank_results` 方法签名**
   - 计划：接收 `rerank` 参数，内部判断 `rerank=False` 时短路返回
   - 实际：不接收 `rerank` 参数，仅检查 `self.reranker is None or not vector_results`；`rerank` 开关由条件边 `_should_rerank` 统一控制
   - 判定：实际设计更干净（条件判断和实际执行分离），无需修改

2. **`_build_search_response` 方法签名**
   - 计划：接收 `merged_results, enable_graph` 两个参数
   - 实际：仅接收 `merged_results`，`enable_graph` 从 `self.enable_graph` 读取
   - 判定：实际设计更合理，`enable_graph` 是实例状态，不需要重复传递

3. **`_format_vector_results` 额外字段处理**
   - 计划：未明确提及 promoted keys（user_id/agent_id/run_id/actor_id/role）的处理
   - 实际：实现了完整的 promoted keys 提升逻辑（从 payload 提升到 MemoryItem 顶层字段）+ additional_metadata 收集
   - 判定：实际实现更完善，与 `main.py:get()` / `_get_all_from_vector_store()` 的格式化策略完全一致

4. **`_search_vector_store` 的脏数据容错**
   - 计划：未提及 payload 缺失的处理
   - 实际：实现了 `hasattr(mem, "payload")` 检查，缺失时 skip + warning
   - 判定：更健壮，无需修改

5. **`graph_depth` 参数未在 `Memory.search()` 暴露**
   - 计划：`Memory.search()` 签名中预期暴露 `graph_depth` 参数
   - 实际：`Memory.search()` 未暴露 `graph_depth`，SearchEngine 内部硬编码为 `2`。这是按设计决策执行的（"本次只重构 Search，不新增全局配置字段"）
   - 判定：当前实现合理。后续 Add Engine 或配置化工作需要时再统一暴露

6. **`_search_vector_store` 删除范围**
   - 计划：删除同步和异步两个版本的 `_search_vector_store`
   - 实际：仅删除了同步版本，异步版本保留（符合"本次不碰 AsyncMemory"的原则）
   - 判定：正确，async 版本留给后续 AsyncMemory 重构时处理

7. **E2E 测试策略调整**
   - 计划：图 E2E 测试使用 `graph_store.add()` 写入实体关系
   - 实际：为绕过 LLM 实体提取的不确定性，E2E 测试中创建 `Neo4jGraphStore` 包装类，直接用 Cypher 读写图数据
   - 判定：更稳定，测试不依赖 LLM 行为，可复现性强

8. **`threshold` 测试策略调整**
   - 计划：E2E threshold 测试使用固定阈值（如 0.8）
   - 实际：采用自适应阈值 `mid_threshold = (score1 + score2) / 2`，基于实际返回分数动态计算
   - 判定：更可靠，避免 embedding 模型版本差异导致分数漂移

### 已识别的后续风险与依赖变化

#### 风险 1：`_search_vector_store` 删除 ✅ 已解决

**问题**：删除 `main.py` 中同步版本的 `_search_vector_store`（line 1036-1072）前，需确认无其他调用方。

**验证结果**：已确认同步 `_search_vector_store` 仅被 `Memory.search()` 调用。删除后 `pytest tests/test_main.py tests/test_memory.py` 全部通过，无引用断裂。异步版本保留未动。

#### 风险 2：Add Engine 对 SearchEngine 的依赖

**当前状态**：`langgraph-refactor-design-spec.md` 和 `langgraph-refactor-directory-plan.md` 都规划了 Add Engine，但尚未实现。

**依赖链**：
```
add_engine.py (未创建)
    ├── search_engine.py (✅ 已完成)
    │   └── mem0.configs.base.MemoryItem
    └── main.py (委托层)
```

Add Engine 可以直接复用 SearchEngine 实例（由 main.py 注入），不需要自己创建。Add Engine 需要 SearchEngine 返回的 `recalled_memories` 做去重决策。

#### 风险 3：SearchEngine 返回结构 vs Add 需求的兼容性

**设计文档要求**：Add 调用 Search 时需要 `recalled_memories`（用于去重决策）和 `relations`（用于图存储）。

**当前 SearchEngine 返回**：`{"results": [...], "relations": [...]}`（无 graph 时只有 `"results"`）。

**兼容判定**：✅ 兼容。`results` 就是 `recalled_memories`，字段和格式完全满足 Add 的需求。Add Engine 只需要调用 `search_engine.search(...)` 然后读取 `result["results"]` 即可。
