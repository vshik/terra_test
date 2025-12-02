# mcp_tool_loader.py

import json
from fastmcp import HTTPClient
from langchain.agents import Tool

MCP_SERVER_URL = "http://mcp-server:8000"

async def load_mcp_tools():
    """Connect to MCP server and load tools as LangChain tools."""
    client = HTTPClient(MCP_SERVER_URL)
    await client.connect()

    discovery = await client.list_tools()
    tools = []

    for tool_def in discovery.tools:
        
        async def _mcp_caller(input_json: str, tool_name=tool_def.name):
            params = json.loads(input_json)
            result = await client.call_tool(tool_name, params)
            return result.content

        tools.append(
            Tool(
                name=tool_def.name,
                func=_mcp_caller,
                description=tool_def.description or f"Calls MCP tool {tool_def.name}"
            )
        )

    return tools
