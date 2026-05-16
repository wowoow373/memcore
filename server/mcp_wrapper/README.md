# mem0 MCP Server — 学习指南

MCP (Model Context Protocol) 包装层，把 mem0 的 12 个 FastAPI REST 端点暴露为 MCP 工具。
MCP Server 作为独立进程运行，通过 `httpx` 把 MCP 调用转发给 FastAPI 后端。

## 架构总览

```
┌─────────────────────────┐
│   LLM (OpenAI/Claude)   │  ← "决策者"：读 tool desc，决定调哪个
│   系统提示词 + tools      │
└───────────┬─────────────┘
            │ Function Calling
┌───────────▼─────────────┐
│   MCP Client (Agent)    │  ← "中间人"：接收 LLM 的 tool_call，
│   client_agent.py        │     通过 MCP 协议转发执行，结果还给 LLM
└───────────┬─────────────┘
            │ streamable-http
            │ http://localhost:8765/mcp
┌───────────▼─────────────┐
│  server/mcp_wrapper/mcp_server  │  ← MCP 协议层：12 个 @mcp.tool()
│  FastMCP + httpx 转发    │     类型注解自动生成 JSON Schema
└───────────┬─────────────┘
            │ HTTP (httpx.AsyncClient)
            │ → http://127.0.0.1:8888
┌───────────▼─────────────┐
│   server/main.py        │  ← REST 协议层：12 个端点
│   FastAPI + Memory SDK   │     Pydantic 校验、Header 鉴权
└─────────────────────────┘
```

## 两个客户端的区别

| 文件 | 角色 | 谁决定调哪个工具？ | 适用场景 |
|------|------|-------------------|---------|
| [client_demo.py](client_demo.py) | 协议测试器 | **你（代码写死）** | 学习 MCP 协议、调试工具 |
| [client_agent.py](client_agent.py) | LLM Agent | **LLM（自主决策）** | 真实的 AI 记忆应用 |

`client_agent.py` 展示了真实 AI 应用的写法：LLM 读取所有工具的描述 → 自主决定何时调用哪个工具 → MCP 执行 → 结果还给 LLM → LLM 生成自然语言回复。理解这个流程是理解 MCP 价值的关键。

## 快速启动

### 前提条件

1. Docker Desktop 已启动（WSL2 集成已开启）
2. `conda` 环境 `mem0` 已配置
3. 端口 8888 和 8765 未被占用

### 第 1 步：启动基础设施（Terminal 1）

```bash
cd server
docker compose up -d    # 启动 postgres + neo4j
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
# 预期返回: {"results":[],"relations":[]}
```

### 第 3 步：启动 MCP Server（Terminal 3）

```bash
cd /home/wowoow/open-source/mem0-main
pip install -r server/mcp_wrapper/requirements.txt
python -m server.mcp.mcp_server
```

### 第 4 步：运行 LLM Agent（Terminal 4）

```bash
# 先确保 OPENAI_API_KEY 等环境变量已设置（server/.env 中）
python server/mcp_wrapper/client_agent.py
```

然后正常对话即可——LLM 会自动决定何时调用记忆工具。

### 第 5 步（可选）：运行协议测试器

```bash
python server/mcp_wrapper/client_demo.py
```

预期输出：`发现 12 个工具` + add/list/search/cleanup 各步骤的 JSON 结果。

## 协议握手手动测试

可以直接用 curl 测试 MCP 协议，不需要任何 MCP 客户端：

```bash
# 1. initialize 握手
curl -i -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
# 预期: 200, serverInfo.name == "mem0-mcp"

# 2. tools/list 查看所有工具
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['result']['tools']]"
# 预期: 列出 12 个工具名

# 3. tools/call 调用 add_memory
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add_memory","arguments":{"messages":[{"role":"user","content":"test"}],"user_id":"curl_test"}}}'
# 预期: 返回 add 结果 JSON
```

## 工具列表

| Tool | HTTP 方法 | FastAPI 端点 | 说明 |
|------|----------|-------------|------|
| `configure` | POST | `/configure` | 动态修改后端配置 |
| `add_memory` | POST | `/memories` | 创建记忆（最复杂） |
| `list_memories` | GET | `/memories` | 列出作用域下所有记忆 |
| `get_memory` | GET | `/memories/{id}` | 按 ID 获取单条记忆 |
| `update_memory` | PUT | `/memories/{id}` | 更新记忆内容 |
| `memory_history` | GET | `/memories/{id}/history` | 查看记忆变更历史 |
| `delete_memory` | DELETE | `/memories/{id}` | 删除单条记忆 |
| `delete_all_memories` | DELETE | `/memories` | 删除作用域下所有记忆 |
| `search_memories` | POST | `/search` | 语义搜索记忆 |
| `start_summary` | POST | `/start_mem_summary` | 触发后台总结生成 |
| `get_summary` | GET | `/get_summary` | 读取最新总结 |
| `reset_all` | POST | `/reset` | 完全重置记忆库 |

## 学习路径

### 第一步：FastAPI 基础 → [server/main.py](../main.py)

只读 **L1-200**，理解：
- `FastAPI()` 实例化：app 的诞生
- `Depends` 依赖注入：鉴权函数如何复用
- `BaseModel` Pydantic 模型：请求体的类型定义
- `@app.post` / `@app.get` 装饰器：端点注册
- `APIKeyHeader` 鉴权：从 HTTP Header 取 API Key

关键代码位置：
- L22: `ADMIN_API_KEY` — 环境变量取 API Key
- L94: `MEMORY_INSTANCE` — 全局单例 Memory 实例
- L96-132: `app = FastAPI(...)` — server 创建
- L134-152: `verify_api_key` — 鉴权依赖函数

### 第二步：FastAPI 端点实现 → [server/main.py](../main.py)

读 **L544-810**，这 260 行定义了 12 个端点。每读一个端点，对照 MCP 工具看对应关系：
- `POST /configure` (L544) → MCP `configure`
- `POST /memories` (L552) → MCP `add_memory`
- `GET /memories` (L588) → MCP `list_memories`
- ...

### 第三步：MCP 参考实现 → [openmemory/api/app/mcp_server.py](../../openmemory/api/app/mcp_server.py)

**只快速浏览以下内容**，其余（SQLAlchemy、contextvars、权限层）跳过：
- L103: `mcp = FastMCP("mem0-mcp-server")` — FastMCP 实例化
- L200: `@mcp.tool(description="...")` — 工具装饰器
- L201: `async def add_memories(text: str) -> str:` — 工具函数签名
- 注意：这个参考实现通过 SSE + FastAPI Router 提供服务，我们用的是 streamable-http 独立进程，更简单。

### 第四步（重点）：MCP Server 实现 → [server/mcp_wrapper/mcp_server.py](mcp_server.py)

**按文件中的编号顺序阅读**，共 7 个部分：
1. **配置常量** — 环境变量注入模式
2. **模块级状态** — 为什么用模块变量存 httpx Client
3. **lifespan** — async context manager 管理连接池
4. **FastMCP 实例** — stateless_http / json_response / host 的含义
5. **工具函数** — `_request` 统一错误处理、`_require_scope` 前置校验
6. **12 个 @mcp.tool()** — 装饰器如 Registration、类型注解 → Schema、docstring → LLM
7. **启动入口** — `mcp.run(transport="streamable-http")`

**对比学习法**：同时打开 `server/main.py` 和本文件，看同一个端点（如 add_memory）在两种协议下的实现差异。

### 第五步（核心）：LLM + MCP Agent → [server/mcp_wrapper/client_agent.py](client_agent.py)

理解 **LLM 决策 + MCP 执行** 的完整协作链：
- `mcp_tools_to_openai_functions()` → MCP tool 定义转 OpenAI Function Calling 格式
- 系统提示词如何引导 LLM 使用记忆工具
- **工具调用循环**：LLM returns tool_calls → MCP executes → results back to LLM → LLM may call more tools → final text response
- `tool_call_id` 的作用：把工具结果和工具调用对应起来
- 自动注入 user_id 的技巧（防止 LLM 忘记传）

关键代码位置：
- L108-128: `mcp_tools_to_openai_functions()` — MCP → OpenAI 格式转换
- L148-165: 系统提示词 — 如何引导 LLM
- L172-190: 工具调用循环 — 整个 Agent 的核心逻辑
- L192-208: MCP 工具执行 — `session.call_tool()`

### 第六步：MCP 协议测试器 → [server/mcp_wrapper/client_demo.py](client_demo.py)

理解 JSON-RPC 三步握手在代码中如何体现：
- `session.initialize()` → JSON-RPC `initialize`
- `session.list_tools()` → JSON-RPC `tools/list`
- `session.call_tool()` → JSON-RPC `tools/call`

同时注意三层 context manager 的嵌套结构。

## openai-agents SDK + MCP + 第三方 LLM 兼容性分析

### 问题来源

`openai-agents`（OpenAI 官方的 Agent SDK）内置了原生 MCP 集成——可以直接把 MCP Server 作为工具源注入 Agent：

```python
agent = Agent(
    name="Assistant",
    mcp_servers=[mcp_server],  # ← MCP Server 直接注入，一行代码
    model="gpt-4o",
)
result = await Runner.run(agent, user_input)
# SDK 自动完成：工具发现 → 格式转换 → LLM 决策 → MCP 执行 → 结果回收
```

但如果 LLM 用的是 DeepSeek、阿里云、Ollama 等第三方提供商，默认配置下会直接报错。原因是 SDK 内部有两个默认行为与第三方 LLM 冲突。

### 社区研究结果

查阅了 [openai-agents-python 官方文档](https://openai.github.io/openai-agents-python/)、GitHub README、以及 models/mcp 两个子系统的 API 参考，总结如下：

#### SDK 架构中影响第三方兼容性的两个关键点

| 默认行为 | 说明 | 第三方兼容性 |
|---------|------|-------------|
| **Responses API** | SDK 默认使用 OpenAI 专有的 `/v1/responses` 端点 | ❌ DeepSeek / 阿里云 / Ollama 全部不支持 |
| **默认 OpenAI Client** | 内置的 `AsyncOpenAI` 指向 `api.openai.com` | ❌ 第三方需要指向自己的 base_url |

#### 社区提供的解决方案

OpenAI 官方在 SDK 中预留了两个切换函数，专门解决这个问题：

```python
from agents import set_default_openai_client, set_default_openai_api

# ① 关掉 Responses API，改用 Chat Completions API（行业标准）
set_default_openai_api("chat_completions")

# ② 把 OpenAI 客户端指向第三方 LLM 的兼容端点
from openai import AsyncOpenAI
custom_client = AsyncOpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1",  # DeepSeek / 阿里云 / Ollama 等
)
set_default_openai_client(custom_client, use_for_tracing=False)
```

这两行配置之后，`Runner.run()` 内部就会走 Chat Completions API 路径，第三方 LLM 完全可用。

另外，SDK 还提供了更细粒度的控制——`OpenAIChatCompletionsModel`，可以 per-agent 指定模型和客户端，与 `MultiProvider` 配合实现混合路由（比如某些 agent 用 OpenAI、某些用 DeepSeek）。

#### MCP 与 SDK 的包名冲突

SDK 内部 `from mcp import ClientSession`，与我们的 `server/mcp/` 目录冲突。已通过**重命名为 `server/mcp_wrapper/`** 解决（"方案 A"）。

### 完整示例：DeepSeek + openai-agents SDK + MCP

```python
import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from agents import (
    Agent, Runner,
    set_default_openai_client,
    set_default_openai_api,
)
from agents.mcp import MCPServerStreamableHttp

# ═══════════════════════════════════════════════════════════
# 配置：把 SDK 切换到 DeepSeek
# ═══════════════════════════════════════════════════════════
api_key = os.getenv("OPENAI_llm_API_KEY")
base_url = os.getenv("OPENAI_llm_URL") + "/v1"
model = os.getenv("OPENAI_llm_Model")   # deepseek-chat

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
set_default_openai_client(client, use_for_tracing=False)
set_default_openai_api("chat_completions")  # ← 核心：不用 Responses API

# ═══════════════════════════════════════════════════════════
# 连接 MCP Server → 注入 Agent → 一行 run()
# ═══════════════════════════════════════════════════════════
async def main():
    async with MCPServerStreamableHttp(
        params={
            "url": "http://localhost:8765/mcp",
            "timeout": 30.0,
        },
        cache_tools_list=True,
    ) as mcp_server:
        agent = Agent(
            name="Memory Assistant",
            instructions=(
                "When the user shares anything about themselves, "
                "use add_memory with user_id='u_demo'."
            ),
            mcp_servers=[mcp_server],  # ← MCP 工具源直接注入
            model=model,
        )

        # 整个 tool calling 循环这一行搞定
        result = await Runner.run(
            agent, "I love dark roast coffee."
        )
        print(result.final_output)

asyncio.run(main())
```

**输出**: LLM 读到 `add_memory` 的 MCP 工具描述 → 自主决定调用 → MCP 执行 → 记忆存入向量库 → LLM 生成自然语言回复。

### SDK 支持的四种 MCP 传输方式

SDK 内置了 4 种 MCP 传输适配器，开箱即用：

| 类名 | 传输方式 | 适用场景 |
|------|---------|---------|
| `MCPServerStreamableHttp` | streamable-http | 跨网络 HTTP 连接（本项目使用） |
| `MCPServerStdio` | stdio | 本地子进程（SDK 负责拉进程） |
| `MCPServerSse` | SSE | HTTP + Server-Sent Events（已弃用） |
| `HostedMCPTool` | 托管 | 工具调用往返在 OpenAI 云端完成（仅 OpenAI 模型） |

```python
# stdio 模式示例：SDK 自动管理子进程
async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    },
) as mcp_server:
    agent = Agent(name="FS Agent", mcp_servers=[mcp_server])
```

### Agent 级别的 MCP 配置

```python
agent = Agent(
    name="Assistant",
    mcp_servers=[mcp_server],
    mcp_config={
        "convert_schemas_to_strict": True,  # MCP 的 JSON Schema 转 strict schema
        "failure_error_function": None,     # None = MCP 工具失败时抛异常
    },
)
```

### 已知限制

| 限制 | 影响 | 应对 |
|------|------|------|
| 默认 5 秒 MCP 工具超时 | `add_memory` 等重操作可能超时 | 调大 `client_session_timeout_seconds` |
| 部分 LLM 不支持 `tool_choice` | 某些 provider 报 `invalid_request_error` | 去掉 `tool_choice` 参数或换模型 |
| 追踪/遥测依赖 OpenAI API | 第三方 API key 无法上传 trace | `use_for_tracing=False` |
| `HostedMCPTool` 仅 OpenAI | 托管 MCP 工具只能在 OpenAI 模型上用 | 用 stdio 或 streamable-http 替代 |
| MCP Server 返回大结果 | 可能撑爆 LLM context window | MCP 层截断或分块处理 |

## MCP Client 生态全景

除了 `openai-agents` SDK，社区中还有多种 MCP 客户端方案。以下是 2025-2026 年主要的 Python 生态选项：

### 生态一览

| 库 / 框架 | 维护方 | 集成方式 | LLM 兼容性 | 代码量 |
|----------|--------|---------|-----------|--------|
| **`mcp` (Python SDK)** | Anthropic 官方 | 手动 `ClientSession` + 自行对接 LLM | 任意（纯协议层，不管 LLM） | ~80 行 |
| **`openai-agents`** | OpenAI 官方 | `Agent(mcp_servers=[...])` 原生 MCP 注入 | 仅 Chat Completions API 提供商 | ~10 行 |
| **`langchain-mcp-adapters`** | LangChain | `load_mcp_tools(session)` → LangChain Tool | 所有 LangChain 支持的 LLM | ~15 行 |
| **`mcp-agent`** | LastMile AI | `MCPAggregator` + `AugmentedLLM` 工作流 | Anthropic / Google / Azure / Bedrock / OpenAI | ~20 行 |
| **Claude Desktop / Claude Code** | Anthropic | JSON 配置 MCP Server 路径 → 自动注入 | Claude 专有 | 0 行代码 |
| **Anthropic SDK 直连** | Anthropic 官方教程 | `ClientSession` + `anthropic.Anthropic().messages.create(tools=...)` | Anthropic Claude | ~50 行 |

### 各方案详解

#### 1. `mcp` Python SDK（官方协议实现）

```python
# 最底层：纯协议封装，不绑定任何 LLM
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8765/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        # ↑ 拿到工具列表后，你需要自己对接 LLM
        result = await session.call_tool("add_memory", {...})
```

**定位**：MCP 协议的 Python 实现，提供 `ClientSession` 作为客户端基元。**不绑定任何 LLM**，你需要自己写 LLM 对接代码（MCP → OpenAI/Anthropic 格式转换 + tool calling 循环）。是其他所有方案的底层依赖。

**适用**：需要完全自由控制 LLM ↔ MCP 对接细节的场景。

#### 2. `openai-agents` SDK（OpenAI 官方 Agent 框架）

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

async with MCPServerStreamableHttp(
    params={"url": "http://localhost:8765/mcp"}
) as mcp_server:
    agent = Agent(
        name="Assistant",
        mcp_servers=[mcp_server],  # ← MCP Server 直接注入
        model="gpt-4o",
    )
    result = await Runner.run(agent, "user message")
```

**定位**：OpenAI 官方的 Agent 框架，内置 MCP 原生集成。MCP Server 可以被直接作为工具源传给 Agent，SDK 自动完成所有的发现、转换、调用、结果回收。

**第三方 LLM 兼容**：需要两行配置（见上文分析章节），适用于所有 OpenAI-compatible API。

**特点**：
- 代码量最少（~10 行核心）
- 自动处理 MCP 工具发现和类型转换
- 支持 4 种 MCP 传输方式（stdio / streamable-http / SSE / 托管）
- 仅支持 OpenAI Function Calling 格式

#### 3. `langchain-mcp-adapters`（LangChain 生态）

```python
# 安装: pip install langchain-mcp-adapters
from langchain_mcp_adapters import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# 单 MCP Server
tools = await load_mcp_tools(session)
agent = create_react_agent("openai:gpt-4o", tools)

# 多 MCP Server（同时连接多个服务）
from langchain_mcp_adapters import MultiServerMCPClient

client = MultiServerMCPClient({
    "math": {"command": "python", "args": ["math_server.py"]},      # stdio
    "weather": {"url": "http://localhost:8000/mcp"},               # streamable-http
})
tools = await client.get_tools()
```

**定位**：LangChain/LangGraph 的 MCP 适配器。把 MCP 工具转成 LangChain Tool 对象，然后可以用在任何 LangChain Agent 中（ReactAgent、ToolNode 等）。

**特点**：
- `load_mcp_tools(session)` 一行转换
- `MultiServerMCPClient` 管理多个 MCP Server
- 支持 stdio 和 streamable-http 两种传输
- 继承了 LangChain 的所有 LLM 提供商支持（OpenAI / Anthropic / Google / 本地模型等）
- 支持运行时 HTTP headers（鉴权/追踪）

#### 4. `mcp-agent`（LastMile AI）

```bash
# 安装
pip install mcp-agent
# 可选 LLM 提供商
pip install "mcp-agent[anthropic,google,azure,bedrock]"
```

```python
# YAML 配置 + CLI 驱动（低代码模式）
# mcp_agent.config.yaml 中定义 MCP Server + Agent + Workflow

# 编程模式
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

aggregator = MCPAggregator(servers=[...])
tools = await aggregator.get_tools()
```

**定位**：专为 MCP 协议设计的 Agent 框架，比 `openai-agents` 更重但更全面。

**特点**：
- **`MCPAggregator`** — 统一管理多个 MCP Server 的工具
- **多 LLM 支持** — Anthropic / Google / Azure / Bedrock / OpenAI
- **内置工作流模式** — Parallel (Map-Reduce)、Router、Orchestrator-Workers、Swarm、Evaluator-Optimizer
- **OAuth 支持** — 内置 OAuth 客户端（GitHub 等）
- **Temporal 持久化** — 可切换为 Temporal 后端，支持暂停/恢复/重试
- **OpenTelemetry** — 内置追踪和 Token 计数
- **MCP 服务暴露** — 可以把 Agent 本身作为 MCP Server 提供给其他客户端

#### 5. Claude Desktop / Claude Code（零代码）

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "mem0": {
      "command": "python",
      "args": ["-m", "server.mcp_wrapper.mcp_server"]
    }
  }
}
```

**定位**：Anthropic 的客户端产品内置 MCP 客户端能力。配置 MCP Server 的路径（stdio 模式）或 URL（HTTP 模式）后，Claude 自动连接并使用。

**特点**：零代码，适合终端用户。但仅限 Claude 模型，不支持自定义。

#### 6. Anthropic SDK 直连（官方教程方案）

MCP 官方文档 [Build an MCP client](https://modelcontextprotocol.io/docs/develop/build-client) 中的教程写法：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic

class MCPClient:
    async def connect_to_server(self, server_script_path: str):
        # ... stdio transport setup ...
        await self.session.initialize()
        tools = await self.session.list_tools()

    async def process_query(self, query: str) -> str:
        # 手动把 MCP tools → Anthropic tools 格式
        available_tools = [{
            "name": t.name, "description": t.description,
            "input_schema": t.inputSchema
        } for t in (await self.session.list_tools()).tools]

        # Anthropic API call
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": query}],
            tools=available_tools,
        )
        # 手动处理 tool_use blocks → session.call_tool() → 结果还给 Claude
```

**定位**：这是 Anthropic 官方教程的写法，和我们的 `client_demo.py` 属于同一级别——纯协议测试器。适合教学，不适合生产。

### 如何选择？

| 场景 | 推荐 |
|------|------|
| **教学/学习 MCP 协议** | `mcp` SDK 直连（`client_demo.py` 就是这个） |
| **生产 Agent（OpenAI 生态）** | `openai-agents` + `mcp_servers=[...]` |
| **生产 Agent（LangChain 生态）** | `langchain-mcp-adapters` + LangGraph |
| **多 LLM 提供商** | `mcp-agent`（支持 Anthropic/Google/Azure/Bedrock） |
| **终端用户直接使用** | Claude Desktop / Claude Code（零代码配置） |
| **完全自由控制** | `mcp` SDK + 手动对接 LLM SDK |

## 核心概念速查

### LLM + MCP 协作流程

```
User: "我每天早上喝拿铁"
       │
       ▼
   LLM 读取 tools 描述:
     - add_memory: "Store new memories..."  ← 匹配！
     - search_memories: "Search..."         ← 不匹配
     - configure: "Set config..."           ← 不匹配
       │
       ▼ 决定：调 add_memory
   LLM 构造参数（根据 inputSchema）:
     {"messages":[{"role":"user","content":"我每天早上喝拿铁"}],"user_id":"u_agent_demo"}
       │
       ▼ MCP tools/call
   MCP Server → httpx → FastAPI → Memory.add() → 返回 {"results":[...]}
       │
       ▼ 结果还给 LLM
   LLM: "好的，我已经记住了你喜欢早上喝拿铁。☕"
```

这就是 MCP 的完整价值闭环：**LLM 决策"做什么"，MCP 负责"怎么做"**。

### MCP → OpenAI Function Calling 转换

```python
# MCP 工具定义（来自 tools/list）
tool = {
    "name": "add_memory",
    "description": "Store new memories...",
    "inputSchema": {                    ← 标准 JSON Schema
        "type": "object",
        "properties": {
            "messages": {"type": "array", ...},
            "user_id":  {"type": "string"}
        }
    }
}

# 转换后（OpenAI Function Calling 格式）
openai_function = {
    "type": "function",
    "function": {
        "name": tool["name"],           ← 直接复用
        "description": tool["description"],  ← 直接复用
        "parameters": tool["inputSchema"]    ← 直接复用，零转换
    }
}
```

同样的转换可以用于 Anthropic Claude（`tool_use` 块）、Google Gemini（`function_declarations`）等。

### 装饰器即注册

```python
# FastAPI 风格
@app.post("/memories")
def add_memory(...): ...

# FastMCP 风格
@mcp.tool()
async def add_memory(...): ...
```

两者设计哲学相同：用装饰器声明 "这是一个接口/工具"，框架负责协议翻译和路由。

### 类型注解 → Schema

```python
# Python 类型                    # 自动生成的 JSON Schema
messages: List[Dict[str,str]]  →  {"type": "array", "items": {"type": "object"}}
user_id: str | None            →  {"type": "string"} (非必填)
limit: int = 100               →  {"type": "integer", "default": 100}
```

FastAPI 用 Pydantic 做同样的事；FastMCP 直接用 Python 原生 type hints。

### 三种 MCP 传输方式

| 传输方式 | 适用场景 | 连接模式 | 备注 |
|---------|---------|---------|------|
| **stdio** | 本地子进程 | 标准输入输出 | Claude Desktop 默认方式 |
| **SSE** | Web 应用 | 长连接 + POST | 需要维持连接 |
| **streamable-http** | 跨网络/跨子系统 | 每次请求独立 HTTP | 本次选择，适合 WSL2→Windows |

### JSON-RPC 2.0 消息格式

```json
// 请求
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add_memory","arguments":{...}}}

// 成功响应
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{...}"}]}}

// 错误响应
{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Tool execution error: ..."}}
```

## WSL2 → Windows 网络说明

WSL2 的 `0.0.0.0` 端口会自动转发到 Windows 同端口。
Windows 上访问 `http://localhost:8765/mcp` 即可命中 WSL 内的 MCP 服务。

如果转发失败（少见），在 PowerShell 中运行：
```powershell
wsl hostname -I   # 获取 WSL2 的 IP
# 然后用 http://<WSL_IP>:8765/mcp
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEM0_BASE_URL` | `http://127.0.0.1:8888` | FastAPI 后端地址 |
| `MEM0_API_KEY` | `my_very_long_custom_key_123456` | FastAPI 鉴权 Key |
| `MEM0_MCP_PORT` | `8765` | MCP Server 监听端口 |

## 常见问题

**Q: 访问 http://localhost:8765/ 返回 404？**
A: MCP 端点挂在 `/mcp` 路径下。正确的地址是 http://localhost:8765/mcp。

**Q: 能加 `--reload` 吗？**
A: 不能。FastMCP 内部有自己的 uvicorn 管理逻辑，`mcp.run()` 不是直接的 uvicorn 命令。修改代码后需要手动重启 MCP Server 进程。

**Q: MCP 层为什么不加鉴权？**
A: WSL2 localhost 端口默认只在本机可达。如果未来要把 8765 暴露到外网，再用 FastMCP 的 ASGI middleware 加 Header 校验。

**Q: stateless_http 和 json_response 有什么区别？**
A: `stateless_http=True` 控制会话模型（每次请求独立，不维护 session）。`json_response=True` 控制响应格式（返回单个 JSON 而非 SSE 流）。两者正交，可以独立配置。

## MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8765/mcp
```

这是官方提供的 MCP 调试工具，可以在浏览器中交互式测试工具调用。
