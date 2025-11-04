import os
import pytest
from fastmcp import Client

pytestmark = pytest.mark.asyncio

@pytest.mark.skipif(os.getenv("RUN_MCP_INTEGRATION", "false").lower() != "true", reason="Integration tests skipped")
async def test_call_clone_repo_tool_via_http():
    async with Client("http://localhost:8100/mcp") as client:
        res = await client.call_tool("clone_repo", {"repo_url": "https://github.com/octocat/Hello-World.git"})
        assert "local_path" in res

@pytest.mark.skipif(os.getenv("RUN_MCP_INTEGRATION", "false").lower() != "true", reason="Integration tests skipped")
async def test_call_get_rightsizing_via_http():
    async with Client("http://localhost:8100/mcp") as client:
        res = await client.call_tool("get_rightsizing_recommendations", {"resource_type":"VM","region":"eastus","limit":1})
        assert "columns" in res and "rows" in res
