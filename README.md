# memcore

基于 [mem0](https://github.com/mem0ai/mem0) 的个人修改版本。

## 当前修改

- **SearchEngine 提取**：将 `Memory.search()` 重构为独立的 `SearchEngine` 类（`mem0/memory/search_engine.py`），支持统一的向量 + 图搜索及可选重排序。
- **LangGraph 重构**：将 `Memory.add()` 流程重构为 LangGraph 状态机。

相关设计文档：

- [langgraph-refactor-design-spec.md](langgraph-refactor-design-spec.md)
- [langgraph-refactor-directory-plan.md](langgraph-refactor-directory-plan.md)
- [search-engine-implementation-plan.md](search-engine-implementation-plan.md)

## 原始项目

- [mem0ai/mem0](https://github.com/mem0ai/mem0)
