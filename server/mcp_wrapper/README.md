# mem0 MCP Wrapper — 双 Server 架构

> 完整部署与接入指南见根目录 [integration-guide.md](../../integration-guide.md)。本文档侧重于 MCP 协议细节和工具内部实现。

MCP (Model Context Protocol) 包装层，把 mem0 的 FastAPI REST 端点暴露为 MCP 工具。支持**工具分离**：标准记忆和流程记忆各自运行在独立的 MCP Server 上，供不同类型的 AI Agent 使用。

## 架构总览

```
   Conversational Agent                Code Agent
   (用户对话、偏好记忆)              (任务执行、经验检索)
         │                                  │
         ▼                                  ▼
┌─────────────────────┐          ┌─────────────────────┐
│ standard_mcp_server │          │ process_mcp_server  │
│   Port 8765         │          │   Port 8766         │
│   12 tools          │          │   2 tools            │
│   add/search/list   │          │   write/search       │
│   update/delete/... │          │   process memory     │
└─────────┬───────────┘          └─────────┬───────────┘
          │                                │
          └────────────┬───────────────────┘
                       │  shared.py (共享工具函数)
                       │  _request, _require_scope, _drop_none
                       │  create_mcp_lifespan
                       │
                       ▼
              ┌─────────────────┐
              │  server/main.py │  ← FastAPI (Port 8888)
              │  Memory SDK     │     同一个 MEMORY_INSTANCE
              └─────────────────┘
```

### 为什么需要两个 Server？

| | Standard MCP | Process MCP |
|---|---|---|
| **目标 Agent** | 对话型（chatbot, assistant） | 代码型（code agent, dev agent） |
| **数据模型** | `{role, content}` 对话消息 | `{Goal, Step, Brief, ...}` 结构化 step |
| **写入时机** | 每句话都可能 | 任务完成后一次性批量写入 |
| **检索模式** | 向量 + 图遍历 | Graph Brief语义 + Chunk Goal + Summary |
| **存储层** | 向量库 + Entity(:Entity) 图 | 向量库(chunk/summary) + Step(:Step) 图 |

两种记忆共享同一个 FastAPI 后端和 Memory SDK，但底层的向量 collection 和 Neo4j 节点标签已经物理隔离。

## 文件清单

| 文件 | 用途 |
|------|------|
| [shared.py](shared.py) | 共享工具函数和配置常量（无状态） |
| [standard_mcp_server.py](standard_mcp_server.py) | 标准记忆 MCP（12 工具，Port 8765） |
| [process_mcp_server.py](process_mcp_server.py) | 流程记忆 MCP（2 工具，Port 8766） |
| [mcp_server.py](mcp_server.py) | 向后兼容别名 → `standard_mcp_server` |
| [client_agent.py](client_agent.py) | 标准记忆 LLM Agent 客户端 |
| [client_demo.py](client_demo.py) | MCP 协议测试器（手动调工具） |

## 快速启动

### 前提

1. Docker Desktop 已启动
2. conda 环境 `mem0` 可用
3. 端口 8888, 8765, 8766 未占用

### 第 1 步：启动基础设施（Terminal 1）

```bash
cd server
docker compose up -d    # postgres + neo4j
```

### 第 2 步：启动 FastAPI 后端（Terminal 2）

```bash
cd server
conda activate mem0
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

验证：
```bash
curl http://localhost:8888/memories?user_id=test \
  -H "X-API-Key: my_very_long_custom_key_123456"
```

### 第 3 步：启动标准记忆 MCP Server（Terminal 3）

```bash
cd /home/wowoow/open-source/mem0-main
python -m server.mcp_wrapper.standard_mcp_server
# 或（向后兼容）
python -m server.mcp_wrapper.mcp_server
# → 监听 http://0.0.0.0:8765/mcp
```

### 第 4 步：启动流程记忆 MCP Server（Terminal 4，可选）

```bash
python -m server.mcp_wrapper.process_mcp_server
# → 监听 http://0.0.0.0:8766/mcp
```

### 第 5 步：运行 Agent

```bash
# 标准记忆 Agent（对话型）
python server/mcp_wrapper/client_agent.py

# 协议测试器
python server/mcp_wrapper/client_demo.py
```

## 工具列表

### Standard MCP（Port 8765, 12 tools）

| Tool | HTTP | FastAPI 端点 | 说明 |
|------|------|-------------|------|
| `configure` | POST | `/configure` | 动态修改后端配置 |
| `add_memory` | POST | `/memories` | 创建记忆 |
| `list_memories` | GET | `/memories` | 列出作用域下所有记忆 |
| `get_memory` | GET | `/memories/{id}` | 按 ID 获取单条记忆 |
| `update_memory` | PUT | `/memories/{id}` | 更新记忆内容 |
| `memory_history` | GET | `/memories/{id}/history` | 查看变更历史 |
| `delete_memory` | DELETE | `/memories/{id}` | 删除单条记忆 |
| `delete_all_memories` | DELETE | `/memories` | 删除作用域下所有记忆 |
| `search_memories` | POST | `/search` | 语义搜索记忆 |
| `start_summary` | POST | `/start_mem_summary` | 触发后台总结 |
| `get_summary` | GET | `/get_summary` | 读取最新总结 |
| `reset_all` | POST | `/reset` | 完全重置记忆库 |

### Process MCP（Port 8766, 2 tools）

| Tool | HTTP | FastAPI 端点 | 说明 |
|------|------|-------------|------|
| `write_process_memory` | POST | `/process-memories` | Flow 1: 写入任务 step summaries |
| `search_process_memory` | POST | `/process-memories/search` | Flow 2: 搜索历史任务经验 |

详细接口设计（参数 Schema、返回结构、错误边界）见 [process_mcp_server.py](process_mcp_server.py) 中各工具的 docstring。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEM0_BASE_URL` | `http://127.0.0.1:8888` | FastAPI 后端地址 |
| `MEM0_API_KEY` | `my_very_long_custom_key_123456` | FastAPI 鉴权 Key |
| `MEM0_MCP_PORT` | `8765` | 标准记忆 MCP 监听端口 |
| `MEM0_PROCESS_MCP_PORT` | `8766` | 流程记忆 MCP 监听端口 |

两个 Server 共享 `MEM0_BASE_URL` 和 `MEM0_API_KEY`（指向同一个 FastAPI 后端）。

## Agent 接入示例

### 标准记忆 Agent（openai-agents SDK）

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

async with MCPServerStreamableHttp(
    params={"url": "http://localhost:8765/mcp"}
) as mcp_server:
    agent = Agent(
        name="Chat Assistant",
        instructions="Use add_memory when the user shares info. Use search_memories before answering.",
        mcp_servers=[mcp_server],
    )
    result = await Runner.run(agent, "I love dark roast coffee.")
```

### 流程记忆 Agent（openai-agents SDK）

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

async with MCPServerStreamableHttp(
    params={"url": "http://localhost:8766/mcp"}
) as mcp_server:
    agent = Agent(
        name="Code Agent",
        instructions=(
            "Before each step, call search_process_memory to find past experience. "
            "After the task completes, call write_process_memory to store the summaries."
        ),
        mcp_servers=[mcp_server],
    )
    result = await Runner.run(agent, "Implement user authentication for this project")
```

### 同时使用两个 MCP Server

```python
async with (
    MCPServerStreamableHttp(params={"url": "http://localhost:8765/mcp"}) as std_mcp,
    MCPServerStreamableHttp(params={"url": "http://localhost:8766/mcp"}) as proc_mcp,
):
    agent = Agent(
        name="Full Agent",
        mcp_servers=[std_mcp, proc_mcp],  # 两个 Server 的工具都可见
        ...
    )
```

## 协议握手手动测试

### Standard MCP

```bash
# initialize
curl -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'

# tools/list
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['result']['tools']]"
# → 列出 12 个工具

# add_memory
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add_memory","arguments":{"messages":[{"role":"user","content":"I prefer dark roast"}],"user_id":"curl_test"}}}'
```

### Process MCP

```bash
# tools/list
curl -s -X POST http://localhost:8766/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['result']['tools']]"
# → 列出 2 个工具: write_process_memory, search_process_memory

# search_process_memory
curl -s -X POST http://localhost:8766/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_process_memory","arguments":{"current_step":{"Goal":"Add auth","Step":"03","Action":"create","Brief":"Create auth.py"},"user_id":"test"}}}'
```

## 错误处理

### 错误层级

三层错误模型，所有工具保持一致：

| 层 | 触发条件 | Agent 收到 |
|----|---------|-----------|
| **MCP 层校验** | scope 参数全为空 | Tools execution error (JSON-RPC `isError: true`) |
| **HTTP 层** | FastAPI 返回 4xx/5xx | `{"error": true, "status": <code>, "detail": "<...>"}` |
| **网络层** | 连接失败/超时 | `{"error": true, "status": null, "detail": "<message>"}` |

### 常见错误字典

| 条件 | 返回 dict |
|------|----------|
| API Key 错误 | `{"error": true, "status": 401, "detail": "Invalid API key."}` |
| Pydantic 校验失败 | `{"error": true, "status": 422, "detail": "[{\"type\": \"missing\", ...}]"}` |
| 后端不可达 | `{"error": true, "status": null, "detail": "All connection attempts failed"}` |
| 流程记忆未配置 | `{"error": true, "status": 500, "detail": "Process memory is not configured..."}` |

所有错误以 dict 形式返回（非异常），LLM 可以阅读 `detail` 字段并自行修正参数重试。

## MCP Inspector

```bash
# 标准记忆
npx @modelcontextprotocol/inspector http://localhost:8765/mcp

# 流程记忆
npx @modelcontextprotocol/inspector http://localhost:8766/mcp
```

## WSL2 -> Windows 网络

WSL2 的 `0.0.0.0` 端口会自动转发到 Windows 同端口。
Windows 上访问 `http://localhost:8765/mcp` 或 `http://localhost:8766/mcp` 即可。

如果转发失败：
```powershell
wsl hostname -I   # 获取 WSL2 IP
# 用 http://<WSL_IP>:8765/mcp
```

## 常见问题

**Q: 访问 http://localhost:8765/ 返回 404？**
A: MCP 端点挂在 `/mcp` 路径下。正确的地址是 http://localhost:8765/mcp。

**Q: 能加 `--reload` 吗？**
A: 不能。FastMCP 内部管理 uvicorn，不支持 `--reload`。修改代码后需要手动重启。

**Q: mcp_server.py 还能用吗？**
A: 可以用。`mcp_server.py` 现在直接 re-export `standard_mcp_server` 的 `mcp` 对象，行为完全一致。`python -m server.mcp_wrapper.mcp_server` 和 `from server.mcp_wrapper.mcp_server import mcp` 都继续工作。

**Q: 两个 Server 能同时运行吗？**
A: 可以。它们监听不同端口（8765 / 8766），各自维护独立的 httpx 连接池，都转发到同一个 FastAPI 后端。

**Q: 流程记忆需要额外配置吗？**
A: 需要在 MemoryConfig 中设置 `process_memory`。如果未配置，调用 process memory 工具会返回 `{"error": true, "status": 500, "detail": "...VALIDATION_005"}`。
