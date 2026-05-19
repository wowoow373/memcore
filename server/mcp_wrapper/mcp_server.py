"""
mem0 MCP Server (backward-compatible entry point)
═══════════════════════════════════════════════════

DEPRECATED: This module is kept for backward compatibility.
It re-exports from standard_mcp_server.py.

New code should import directly:
  from server.mcp_wrapper.standard_mcp_server import mcp  # standard memory
  from server.mcp_wrapper.process_mcp_server import mcp   # process memory

The original 12-tool MCP server with detailed learning-oriented
comments lives in standard_mcp_server.py now. This file
simply aliases it so that existing integrations (client_agent.py,
client_demo.py, Claude Desktop configs referencing this module)
continue working without changes.

Run:
  python -m server.mcp_wrapper.mcp_server
    → runs the standard memory MCP server on port 8765
"""

from server.mcp_wrapper.standard_mcp_server import mcp

__all__ = ["mcp"]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
