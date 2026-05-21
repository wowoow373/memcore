# memcore

基于 [mem0](https://github.com/mem0ai/mem0) 的个人修改版本。

## 当前修改

### LangGraph 重构：标准记忆引擎

将 `Memory.add()` 和 `Memory.search()` 的内联逻辑拆分为独立的 LangGraph 引擎：

- **[AddEngine](mem0/memory/add_engine.py)** — 8 节点 2 条件分支的写入编排。实现 "add-with-search-back" 模式：写入前先调用 SearchEngine 召回已有记忆，LLM 对比决策 ADD/UPDATE/DELETE/NONE，返回写入结果 + 联想记忆。
- **[SearchEngine](mem0/memory/search_engine.py)** — 6 节点 2 条件分支的统一召回。向量搜索 + 图遍历 + 合并去重 + 可选 rerank。
- **[MemoryGraph.ingest()](mem0/memory/graph_memory.py#L888-L1028)** — 图写入统一接口。接收上游预提取的实体/关系/删除项，不调用 LLM。

### 流程记忆（Process Memory）

新增三层颗粒度的任务执行记忆系统，用于 Code Agent 的经验沉淀与联想：

- **[ProcessMemoryAddEngine](mem0/memory/process_add_engine.py)** — Flow 1（任务完成后）写入编排。5 节点 LangGraph：preprocess → search → decide → execute → assemble。三层写入：Graph（Step 节点 + DEPENDS_ON 边）、Chunk（Goal 级向量）、Summary（完整链路向量）。
- **[ProcessMemorySearchEngine](mem0/memory/process_search_engine.py)** — Flow 1 去重 + Flow 2（任务进行中）检索。纯只读，无 LLM，无 LangGraph。Brief embedding 语义匹配 + 前一步语义筛选 + Goal/Summary 向量召回。
- **[MemoryGraph.search_nodes_by_embedding()](mem0/memory/graph_memory.py#L153-L218)** — Step 节点的 Brief embedding 语义检索，Cypher 层 cosine 相似度匹配。
- **[ProcessMemoryConfig](mem0/configs/base.py#L73-L99)** — 流程记忆独立配置：`process_memory.vector_store` + `process_memory.graph_store` + 超参数。
- **`Memory.search_process()`** — Flow 2 公开入口。

### 配置扩展

- `MemoryConfig.process_memory` 可选字段，设为 `ProcessMemoryConfig` 后启用流程记忆。
- `Memory.add(memory_type="process_memory")` 路由到 ProcessMemoryAddEngine。

## 设计文档

- [memory-api-spec.md](memory-api-spec.md) — 完整接口规格书（标准记忆 + 流程记忆的 Search/Add/Back 全周期）
- [langgraph-refactor-design-spec.md](archive/designs/langgraph-refactor-design-spec.md) — 标准记忆 LangGraph 重构设计
- [add-engine-design.md](archive/designs/add-engine-design.md) — AddEngine 设计规格
- [process-memory-design.md](archive/designs/process-memory-design.md) — 流程记忆整体设计
- [process-add-engine-design.md](archive/designs/process-add-engine-design.md) — ProcessMemoryAddEngine 详细设计
- [process-search-engine-design.md](archive/designs/process-search-engine-design.md) — ProcessMemorySearchEngine 详细设计
- [lucky-discovering-dahl.md](archive/designs/lucky-discovering-dahl.md) — GraphStore 与 LLM 实体提取职责拆分
- [search-engine-implementation-plan.md](archive/plans/search-engine-implementation-plan.md) — SearchEngine 重构实施计划
- [add-engine-implementation.md](archive/plans/add-engine-implementation.md) — AddEngine 实现文档
- [langgraph-refactor-directory-plan.md](archive/plans/langgraph-refactor-directory-plan.md) — LangGraph 重构目录结构规划
- [process-memory-directory-plan.md](archive/plans/process-memory-directory-plan.md) — Process Memory 目录结构

## 原始项目

- [mem0ai/mem0](https://github.com/mem0ai/mem0)
