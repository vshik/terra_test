# wrappers/mcp_wrappers.py
import asyncio
from fastmcp import Client

MCP_URL = "http://mcp-server:9000/mcp"  # or env var

async def mcp_get_rightsizing(environment, app_id):
    async with Client(MCP_URL) as client:
        return await client.call_tool("get_rightsizing_recommendations", {
            "environment": environment,
            "app_id": app_id
        })

async def mcp_clone_repo(repo_url):
    async with Client(MCP_URL) as client:
        return await client.call_tool("clone_repo", {"repo_url": repo_url})

async def mcp_analyze_infra(repo_url):
    async with Client(MCP_URL) as client:
        return await client.call_tool("analyze_infra", {"repo_url": repo_url})

async def mcp_update_infra(repo_url, analysis, recommendations):
    async with Client(MCP_URL) as client:
        return await client.call_tool("update_infra", {
            "repo_url": repo_url,
            "analysis": analysis,
            "recommendations": recommendations
        })
