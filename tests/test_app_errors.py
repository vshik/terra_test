import pytest
from unittest.mock import AsyncMock, patch
import app as app_module

@pytest.mark.asyncio
@patch("app.ask_llm", new_callable=AsyncMock)
async def test_orchestrate_handles_invalid_llm_output(mock_ask):
    # LLM returns garbage
    mock_ask.return_value = "I am not JSON"
    res = app_module.safe_parse_llm_output("I am not JSON")
    assert isinstance(res, dict)
    assert res["tool"] == "none"

@pytest.mark.asyncio
@patch("app.call_mcp_tool", new_callable=AsyncMock)
@patch("app.ask_llm", new_callable=AsyncMock)
async def test_call_mcp_tool_connection_failure(mock_ask, mock_call):
    mock_ask.return_value = '{"tool":"clone_repo","params":{"repo_url":"x"}}'
    mock_call.side_effect = Exception("Connection failed")
    with pytest.raises(Exception):
        await app_module.orchestrate("clone repo")

@pytest.mark.asyncio
@patch("mcp_server.subprocess.run", side_effect=Exception("git failed"))
def test_clone_repo_git_failure(mock_subproc):
    from mcp_server import CloneInput, clone_repo
    params = CloneInput(repo_url="https://github.com/org/repo.git")
    with pytest.raises(Exception):
        clone_repo.fn(params, None)

@pytest.mark.asyncio
@patch("mcp_server.pyodbc.connect", side_effect=Exception("db down"))
def test_get_rightsizing_db_failure(mock_connect):
    from mcp_server import FinOpsQueryInput, get_rightsizing_recommendations
    params = FinOpsQueryInput(resource_type="VM", region="eastus")
    with pytest.raises(Exception):
        get_rightsizing_recommendations.fn(params, None)

@pytest.mark.asyncio
@patch("mcp_server.GITHUB_CLIENT.get_repo", side_effect=Exception("gh api failure"))
def test_create_pr_github_failure(mock_get_repo):
    from mcp_server import PRInput, create_pull_request
    params = PRInput(repo_fullname="org/repo", source_branch="feat", title="t")
    with pytest.raises(Exception):
        create_pull_request.fn(params, None)
