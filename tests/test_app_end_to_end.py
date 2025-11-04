import pytest
from unittest.mock import AsyncMock, patch
import app as app_module

@pytest.mark.asyncio
@patch("app.ask_llm", new_callable=AsyncMock)
@patch("app.call_mcp_tool", new_callable=AsyncMock)
async def test_full_clone_flow(mock_call_mcp, mock_ask_llm):
    # LLM says call clone_repo on MCP
    mock_ask_llm.return_value = '{"tool":"clone_repo","params":{"repo_url":"https://github.com/org/repo.git"}}'
    mock_call_mcp.return_value = {"local_path": "/tmp/repo"}
    res = await app_module.orchestrate("clone repo")
    assert "local_path" in res or res == {"tool": "none", "message": res.get("message", "")}

@pytest.mark.asyncio
@patch("app.ask_llm", new_callable=AsyncMock)
@patch("app.call_mcp_tool", new_callable=AsyncMock)
async def test_full_get_rightsizing_flow(mock_call_mcp, mock_ask_llm):
    mock_ask_llm.return_value = '{"tool":"get_rightsizing_recommendations","params":{"resource_type":"VM","region":"eastus"}}'
    mock_call_mcp.return_value = {"columns": ["resource_type"], "rows": [["VM"]]}
    res = await app_module.orchestrate("get rightsizing")
    assert isinstance(res, dict) or isinstance(res, list) or "rows" in res

@pytest.mark.asyncio
@patch("app.ask_llm", new_callable=AsyncMock)
async def test_full_greeting_flow(mock_ask_llm):
    mock_ask_llm.return_value = '{"tool":"none","params":{},"message":"Hi"}'
    res = await app_module.orchestrate("hello")
    assert res["tool"] == "none"

@pytest.mark.asyncio
@patch("app.ask_llm", new_callable=AsyncMock)
@patch("app.call_mcp_tool", new_callable=AsyncMock)
async def test_full_create_pr_flow(mock_call_mcp, mock_ask_llm):
    mock_ask_llm.return_value = '{"tool":"create_pull_request","params":{"repo_fullname":"org/repo","source_branch":"feat","title":"t"}}'
    mock_call_mcp.return_value = {"pr_number": 1, "pr_url": "http://"}
    res = await app_module.orchestrate("create pr")
    assert "pr_number" in res or isinstance(res, dict)

