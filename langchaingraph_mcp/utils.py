# orchestrator/utils.py
import asyncio
from fastmcp import Client

async def call_mcp_tool(mcp_url: str, tool_name: str, params: dict):
    """Call a tool on any MCP server asynchronously."""
    async with Client(mcp_url) as client:
        response = await client.call_tool(tool_name, params)
        return response
