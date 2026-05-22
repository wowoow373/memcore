# 部署与接入指南

外部 Agent / 应用通过 **MCP (Model Context Protocol)** 接入 mem0 记忆服务。MCP 层（`server/mcp_wrapper/`）将 mem0 的 FastAPI REST 端点包装为 MCP 工具，供 AI Agent SDK（如 openai-agents）直接消费。

## 架构总览

```
  Conversational Agent              Code Agent
  (对话型、偏好记忆)               (代码型、经验检索)
        │                                │
        ▼                                ▼
┌──────────────────────┐       ┌──────────────────────┐
│ standard_mcp_server  │       │ process_mcp_server   │
│   Port 8765          │       │   Port 8766          │
│   12 tools           │       │   2 tools             │
│   add/search/list/...│       │   write/search        │
└─────────┬────────────┘       └─────────┬────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │  httpx HTTP 转发
                         ▼
              ┌────────────────────┐
              │  FastAPI (Port 8888)│
              │  server/main.py    │
              │  Memory SDK        │
              └────────┬───────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌──────────┐            ┌──────────┐
   │ pgvector │            │  Neo4j   │
   │ :8432    │            │  :8687   │
   └──────────┘            └──────────┘
```

- **FastAPI**（port 8888）：核心记忆引擎，封装 Memory SDK 的全部操作。流程记忆通过配置中的 `process_memory` 字段启用，使用独立的 pgvector collection 和 `:Step` 节点标签与标准记忆物理隔离。
- **Standard MCP**（port 8765）：供对话型 Agent 使用，暴露 12 个工具（add/search/list/update/delete 等）。
- **Process MCP**（port 8766）：供代码型 Agent 使用，暴露 2 个工具（write_process_memory / search_process_memory），对应 Flow 1 写入和 Flow 2 检索。

## 快速启动

### 前提

- Docker 已启动
- Python 3.10+（推荐 conda 环境 `mem0`）
- 端口 8888 / 8765 / 8766 未被占用

### 第 1 步：启动基础设施

```bash
cd server
docker compose up -d
```

启动后验证：

```bash
docker compose ps
# postgres 和 neo4j 状态均为 healthy
```

### 第 2 步：配置环境变量

编辑 `server/.env`：

```bash
# ── LLM（事实提取 + 决策）──
OPENAI_llm_API_KEY=sk-xxx
OPENAI_llm_Model=deepseek-chat
OPENAI_llm_URL=https://api.deepseek.com

# ── Embedding（向量相似度）──
OPENAI_EMBEDDER_API_KEY=sk-xxx
OPENAI_EMBEDDER_MODEL=text-embedding-v4
OPENAI_EMBEDDER_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ── 图数据库 ──
NEO4J_URI=bolt://localhost:8687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=mem0graph

# ── 向量数据库 ──
POSTGRES_HOST=localhost
POSTGRES_PORT=8432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# ── API 鉴权（生产环境请替换为强密码）──
ADMIN_API_KEY=my_very_long_custom_key_123456
```

Docker 端口映射：
| 服务 | 宿主机端口 | 容器端口 |
|------|-----------|----------|
| pgvector | 8432 | 5432 |
| Neo4j Bolt | 8687 | 7687 |
| Neo4j HTTP | 8474 | 7474 |

### 第 3 步：启动 FastAPI 后端

```bash
cd server
conda activate mem0
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

验证：

```bash
curl http://localhost:8888/memories?user_id=test \
  -H "X-API-Key: my_very_long_custom_key_123456"
# → {"results": []}
```

Swagger 文档：http://localhost:8888/docs

### 第 4 步：启动 MCP Server

```bash
# 终端 A：标准记忆 MCP（对话型 Agent 用）
cd /home/wowoow/open-source/mem0-main
python -m server.mcp_wrapper.standard_mcp_server
# → 监听 http://0.0.0.0:8765/mcp

# 终端 B：流程记忆 MCP（代码型 Agent 用，可选）
python -m server.mcp_wrapper.process_mcp_server
# → 监听 http://0.0.0.0:8766/mcp
```

MCP Server 的环境变量覆盖（可选）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEM0_BASE_URL` | `http://127.0.0.1:8888` | FastAPI 后端地址 |
| `MEM0_API_KEY` | `my_very_long_custom_key_123456` | 鉴权 Key |
| `MEM0_MCP_PORT` | `8765` | 标准 MCP 监听端口 |
| `MEM0_PROCESS_MCP_PORT` | `8766` | 流程 MCP 监听端口 |

## 外部接入

### 方式一：openai-agents SDK（推荐）

```python
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.mcp import MCPServerStreamableHttp

async def main():
    # 配置 LLM 后端（DeepSeek / 阿里云 / Ollama 等 OpenAI-compatible 均可）
    client = AsyncOpenAI(
        api_key="sk-xxx",
        base_url="https://api.deepseek.com/v1",
    )
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")

    async with MCPServerStreamableHttp(
        params={"url": "http://localhost:8765/mcp"},
    ) as mcp_server:
        agent = Agent(
            name="Assistant",
            instructions=(
                "You have persistent memory. "
                "Use add_memory when the user shares info about themselves. "
                "Use search_memories before answering personal questions."
            ),
            mcp_servers=[mcp_server],
            model="deepseek-chat",
        )
        result = await Runner.run(agent, "I love dark roast coffee.")
        print(result.final_output)

asyncio.run(main())
```

完整示例见 [server/mcp_wrapper/client_agent.py](server/mcp_wrapper/client_agent.py)。

### 方式二：同时使用两个 MCP Server

```python
async with (
    MCPServerStreamableHttp(params={"url": "http://localhost:8765/mcp"}) as std_mcp,
    MCPServerStreamableHttp(params={"url": "http://localhost:8766/mcp"}) as proc_mcp,
):
    agent = Agent(
        name="Full Agent",
        mcp_servers=[std_mcp, proc_mcp],
        ...
    )
```

### 方式三：MCP Inspector（调试用）

```bash
npx @modelcontextprotocol/inspector http://localhost:8765/mcp
npx @modelcontextprotocol/inspector http://localhost:8766/mcp
```

### 方式四：HTTP 直连 FastAPI（不使用 MCP）

如果不需要 MCP 协议封装，直接将 FastAPI 作为 REST API 使用：

```bash
# 创建记忆
curl -X POST http://localhost:8888/memories \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my_very_long_custom_key_123456" \
  -d '{"messages":[{"role":"user","content":"My name is John"}],"user_id":"u1"}'

# 语义搜索
curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my_very_long_custom_key_123456" \
  -d '{"query":"What is my name?","user_id":"u1"}'
```

完整 REST API 文档见 http://localhost:8888/docs。

## 工具清单

### Standard MCP（Port 8765, 12 tools）

| 工具名 | HTTP | FastAPI 端点 | 说明 |
|--------|------|-------------|------|
| `configure` | POST | `/configure` | 热更新后端配置 |
| `add_memory` | POST | `/memories` | 创建记忆。标准模式：LLM 提取事实 → 去重决策 → 写入 |
| `list_memories` | GET | `/memories` | 列出作用域下所有记忆 |
| `get_memory` | GET | `/memories/{id}` | 按 ID 获取单条记忆 |
| `update_memory` | PUT | `/memories/{id}` | 更新记忆内容 |
| `memory_history` | GET | `/memories/{id}/history` | 查看变更历史 |
| `delete_memory` | DELETE | `/memories/{id}` | 删除单条记忆 |
| `delete_all_memories` | DELETE | `/memories` | 删除作用域下所有记忆 |
| `search_memories` | POST | `/search` | 语义搜索（向量 + 可选图遍历 + 可选 rerank）|
| `start_summary` | POST | `/start_mem_summary` | 触发后台总结 |
| `get_summary` | GET | `/get_summary` | 读取最新总结 |
| `reset_all` | POST | `/reset` | 完全重置记忆库 |

### Process MCP（Port 8766, 2 tools）

| 工具名 | HTTP | FastAPI 端点 | 说明 |
|--------|------|-------------|------|
| `write_process_memory` | POST | `/process-memories` | Flow 1：任务完成后写入 step summaries 到三层存储 |
| `search_process_memory` | POST | `/process-memories/search` | Flow 2：任务执行前搜索历史经验（只读）|

`write_process_memory` 的输入是结构化的 step summary 数组：

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

内部 5 节点 LangGraph 流水线：preprocess → search → decide → execute → assemble。三层存储：
- **Graph**（`:Step` 节点 + `DEPENDS_ON` 边）
- **Chunk**（Goal 粒度向量，`memory_type="process_chunk"`）
- **Summary**（完整链路向量，`memory_type="process_summary"`）

## 错误处理

三层错误模型：

| 层级 | 触发条件 | 返回 |
|------|---------|------|
| MCP 层校验 | scope 参数全为空 | `Tool execution error`（JSON-RPC `isError: true`）|
| HTTP 层 | FastAPI 返回 4xx/5xx | `{"error": true, "status": <code>, "detail": "<message>"}` |
| 网络层 | MCP → FastAPI 连接失败 | `{"error": true, "status": null, "detail": "<message>"}` |

常见错误：

| 条件 | 返回内容 |
|------|---------|
| API Key 错误 | `{"error": true, "status": 401, "detail": "Invalid API key."}` |
| scope 缺失 | 工具抛出 `ValueError`，MCP 层返回 JSON-RPC error |
| 流程记忆未配置 | `{"error": true, "status": 500, "detail": "...VALIDATION_005"}` |
| 后端不可达 | `{"error": true, "status": null, "detail": "All connection attempts failed"}` |

LLM 可以阅读错误 dict 中的 `detail` 字段并自行修正参数重试。

## 文件清单

| 文件 | 用途 |
|------|------|
| `server/main.py` | FastAPI 后端，封装 Memory SDK |
| `server/mcp_wrapper/shared.py` | MCP 共享函数（配置常量、HTTP 转发、scope 校验、lifespan 工厂）|
| `server/mcp_wrapper/standard_mcp_server.py` | 标准记忆 MCP Server（12 tools, Port 8765）|
| `server/mcp_wrapper/process_mcp_server.py` | 流程记忆 MCP Server（2 tools, Port 8766）|
| `server/mcp_wrapper/mcp_server.py` | 向后兼容别名 → standard_mcp_server |
| `server/mcp_wrapper/client_agent.py` | 标准记忆 LLM Agent 客户端（openai-agents SDK 集成示例）|
| `server/mcp_wrapper/client_demo.py` | MCP 协议测试器（手动调 tools）|
| `server/.env` | 环境变量配置 |
| `server/docker-compose.yaml` | PostgreSQL (pgvector) + Neo4j |

## 设计文档

- [memory-api-spec.md](memory-api-spec.md) — 完整接口规格书（标准记忆 + 流程记忆的 Search/Add/Back 全周期）
