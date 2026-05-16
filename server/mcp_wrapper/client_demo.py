"""
MCP 协议测试器 — 手动验证 MCP Server 的工具是否正常工作
═══════════════════════════════════════════════════════════════

⚠️ 想看「LLM 自动决定调哪个工具」的真实 AI 应用写法？
   请看同目录下的 [client_agent.py](client_agent.py)。

【文件定位】
  这个文件是"MCP 协议测试器"——手动指定工具名和参数，
  用来验证 MCP Server 的工具注册和透传链路是否正确。
  相当于用 Python 代码代替 curl 来做协议级测试。

  真实 AI 应用（如 client_agent.py）不会手动指定工具
  ——而是让 LLM 读取工具描述后自主决定调哪个。
  本文件的价值在于暴露 MCP 协议的底层细节。

【MCP 客户端的三步握手协议】

  所有 MCP 通信都遵循 JSON-RPC 2.0 规范。一个典型的 MCP 会话分为：

  ┌──────────────────────────────────────────────────────────┐
  │ 步骤 1: initialize（初始化）                              │
  │   Client → Server:  "我支持 protocolVersion=2024-11-05"  │
  │   Server → Client:  "我是 mem0-mcp，我支持 tools/prompts" │
  │   这一步协商协议版本和能力集。                              │
  ├──────────────────────────────────────────────────────────┤
  │ 步骤 2: tools/list（发现工具）                            │
  │   Client → Server:  "列出所有可用工具"                    │
  │   Server → Client:  [{name, description, inputSchema}...]│
  │   这一步让 Client 知道有哪些工具、每个工具需要什么参数。     │
  ├──────────────────────────────────────────────────────────┤
  │ 步骤 3: tools/call（调用工具）                            │
  │   Client → Server:  "调 add_memory, 参数是 {...}"        │
  │   Server → Client:  {"results": [...]}                   │
  │   这是真正执行业务操作的一步，可以反复执行。                │
  └──────────────────────────────────────────────────────────┘

【streamable-http 传输方式】

  不同于 stdio（标准输入输出，适合本地子进程）或 SSE（Server-Sent Events），
  streamable-http 是双向 HTTP POST 模式：
    - 每条 JSON-RPC 消息都是一个独立的 HTTP POST 请求
    - 不需要维持长连接
    - 可以直接用 curl 测试（见 README.md 中的手动测试命令）

【Python Client API 的层次】

  mcp 库提供了 3 层抽象来写客户端：

  第 1 层（最底层）：mcp.client.streamable_http.streamablehttp_client
    → 负责建立 TCP 连接，收发原始 JSON-RPC 字节流
    → 返回 (read_stream, write_stream, _) 三个对象
    → read_stream:  内存管道，Server → Client 的数据从这里读
    → write_stream: 内存管道，Client → Server 的数据往这里写

  第 2 层（中间层）：mcp.ClientSession
    → 包装 read/write stream，提供结构化方法
    → session.initialize()  → 发送 initialize 请求（步骤 1）
    → session.list_tools()  → 发送 tools/list 请求（步骤 2）
    → session.call_tool()   → 发送 tools/call 请求（步骤 3）
    → 自动处理 JSON-RPC 的消息 ID、错误码等协议细节

  第 3 层（最高层）：FastMCP Client（本文件未使用）
    → 进一步封装，提供更简洁的 API
    → 适合生产环境，但不利于学习协议细节

【阅读建议】
  1. 先看 main() 的整体结构 — 一个 async 函数，贯穿整个会话
  2. 理解三层 context manager 的嵌套关系
  3. 看每个 call_tool 的参数格式 — 注意参数名必须和 mcp_server.py 的一致
  4. 最后看 negative test — 理解工具内的校验如何反馈给客户端
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
    """
    MCP 客户端主函数 — 完整的 "连接 → 发现 → 调用 → 清理" 流程。
    每一步都对应 MCP 协议中的一个 JSON-RPC 交互。
    """

    # ═══════════════════════════════════════════════════════════════════════
    # 第一步：建立 streamable-http 连接
    # ═══════════════════════════════════════════════════════════════════════
    # streamablehttp_client() 返回一个 async context manager。
    # 进入 async with 块时，它建立到 MCP Server 的 HTTP 连接。
    # 离开 async with 块时，自动关闭连接。
    #
    # 返回值是 (read_stream, write_stream, _) 三元组：
    #   read_stream  — MemoryObjectStream，从 Server 读响应消息
    #   write_stream — MemoryObjectStream，向 Server 写请求消息
    #   _            — 连接元信息（本 demo 用不到，用 _ 忽略）
    #
    # 【URL 必须是 /mcp 路径】
    #   FastMCP 的 streamable-http transport 默认把端点挂在 /mcp。
    #   如果访问 http://localhost:8765/ （根路径），会返回 404。
    #   这是新手最容易踩的坑。

    async with streamablehttp_client("http://localhost:8765/mcp") as (read, write, _):

        # ═══════════════════════════════════════════════════════════════════
        # 第二步：创建 MCP 会话
        # ═══════════════════════════════════════════════════════════════════
        # ClientSession 包装了 read/write stream，向上层提供
        # 类型安全的工具调用方法。
        #
        # session.initialize() → 发送 JSON-RPC initialize 请求。
        #   这是 MCP 协议规定的"第一件事"。Server 会返回它的名称、
        #   版本、以及它支持的能力（tools、prompts、resources 等）。
        #   如果不先 initialize 就调 list_tools，会收到协议错误。

        async with ClientSession(read, write) as session:
            await session.initialize()

            # ───────────────────────────────────────────────────────────────
            # 步骤 A：发现工具 (tools/list)
            # ───────────────────────────────────────────────────────────────
            # 这是在问 MCP Server："你能做什么？"
            # Server 返回一个 Tool 列表，每个工具包含：
            #   - name: 工具名（如 "add_memory"）
            #   - description: 描述文档（即 @mcp.tool() 装饰的函数 docstring）
            #   - inputSchema: JSON Schema（由类型注解自动生成）
            #
            # 【关键理解】description 是写给 LLM 看的，不是写给人看的。
            #   当真实的 AI 客户端（如 Claude Desktop）连接时，
            #   它会把这些 description 注入到 LLM 的系统提示词中，
            #   LLM 根据 description 决定什么时候调用哪个工具。
            #   所以 mcp_server.py 里的 docstring 写法很重要：
            #   必须包含 "When to use" 和示例。

            tools = await session.list_tools()
            print(f"发现 {len(tools.tools)} 个工具:")
            for t in tools.tools:
                # 只显示前 80 个字符的简介
                print(f"  - {t.name}: {t.description[:80]}...")

            # ───────────────────────────────────────────────────────────────
            # 步骤 B：调用工具 (tools/call) — 核心交互
            # ───────────────────────────────────────────────────────────────
            # session.call_tool(name, arguments) 发送 JSON-RPC tools/call。
            #
            # 底层实际发送的 JSON-RPC 消息类似于：
            #   {
            #     "jsonrpc": "2.0",
            #     "id": 2,
            #     "method": "tools/call",
            #     "params": {
            #       "name": "add_memory",
            #       "arguments": {
            #         "messages": [{"role":"user","content":"..."}],
            #         "user_id": "u_demo_mcp"
            #       }
            #     }
            #   }
            #
            # 【返回值结构】
            #   call_tool() 返回 CallToolResult，其中 .content 是一个列表，
            #   每个元素是 ContentBlock。对于 json_response=True 的 Server，
            #   content[0].text 就是工具函数返回的 JSON 字符串。

            uid = "u_demo_mcp"  # 测试用的 user_id

            # --- add_memory: 创建记忆 ---
            # 参数名必须和 mcp_server.py 中 add_memory 的函数参数名完全一致。
            # messages 是一个列表，每个元素有 role 和 content 两个字段。
            print("\n--- add_memory ---")
            r1 = await session.call_tool("add_memory", {
                "messages": [{"role": "user", "content": "I love dark roast coffee."}],
                "user_id": uid,
            })
            # content[0].text 是 JSON 字符串（因为 Server 端 json_response=True）
            text1 = r1.content[0].text if r1.content else str(r1)
            print(text1[:300])

            # --- list_memories: 列出所有记忆 ---
            # 用来验证 add 是否成功
            print("\n--- list_memories ---")
            r2 = await session.call_tool("list_memories", {"user_id": uid})
            text2 = r2.content[0].text if r2.content else str(r2)
            print(text2[:300])

            # --- search_memories: 语义搜索 ---
            # 用自然语言查询相关记忆。这里搜 "coffee preferences"，
            # 应该能找回刚才存的 "dark roast coffee"。
            print("\n--- search_memories ---")
            r3 = await session.call_tool("search_memories", {
                "query": "coffee preferences",
                "user_id": uid,
                "limit": 3,
            })
            text3 = r3.content[0].text if r3.content else str(r3)
            print(text3[:300])

            # --- delete_all_memories: 清理 ---
            # 测试完删除所有数据，保持环境干净
            print("\n--- delete_all_memories ---")
            r4 = await session.call_tool("delete_all_memories", {"user_id": uid})
            text4 = r4.content[0].text if r4.content else str(r4)
            print(text4[:200])

            # ───────────────────────────────────────────────────────────────
            # 步骤 C：负面测试 — 不传 scope 参数
            # ───────────────────────────────────────────────────────────────
            # 调 add_memory 但故意不传 user_id/agent_id/run_id 中的任何一个。
            # 预期结果：MCP 层直接报错，消息中提示需要 identifier。
            #
            # 【这个测试验证了什么？】
            #   mcp_server.py 中的 _require_scope() 函数，
            #   它在 HTTP 请求发出之前就拦截了不合法的调用。
            #   如果这个校验不存在，请求会发到 FastAPI，
            #   FastAPI 返回 400，MCP 再包装成错误——多一圈延迟。
            #
            # 【注意异常类型】
            #   call_tool 抛异常的情况说明 Server 端的工具函数抛了异常
            #   （这里是 ValueError）。如果 Server 端返回了 dict（即使含 error），
            #   这里不会抛异常，而是正常返回 CallToolResult。

            print("\n--- 负面测试: add_memory 不传 scope ---")
            try:
                r5 = await session.call_tool("add_memory", {
                    "messages": [{"role": "user", "content": "test"}],
                })
                text5 = r5.content[0].text if r5.content else str(r5)
                print(f"结果: {text5[:200]}")
            except Exception as e:
                print(f"错误（符合预期）: {e}")

            print("\n=== 所有检查通过 ===")


# ═══════════════════════════════════════════════════════════════════════════════
# Python asyncio 启动入口
# ═══════════════════════════════════════════════════════════════════════════════
# asyncio.run(main()) 是 Python 3.7+ 的标准写法。
# 它创建事件循环 → 运行 main 协程 → 清理事件循环。
# 等价于旧版写法：
#   loop = asyncio.get_event_loop()
#   loop.run_until_complete(main())
#   loop.close()

if __name__ == "__main__":
    asyncio.run(main())
