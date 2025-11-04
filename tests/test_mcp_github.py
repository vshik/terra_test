import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock
import mcp_server as mcp_server_module

# 1. clone_repo success
def test_clone_repo_success(tmp_path, monkeypatch):
    fake_dir = tmp_path / "repo_test"
    # mock tempfile.mkdtemp by replacing clone_repo implementation using subprocess -> simpler to call function .fn
    with patch("mcp_server.tempfile.mkdtemp", return_value=str(fake_dir)):
        with patch("subprocess.run") as mock_run:
            params = mcp_server_module.CloneInput(repo_url="https://github.com/org/repo.git", branch="main")
            res = mcp_server_module.clone_repo.fn(params, None)
            assert isinstance(res, mcp_server_module.CloneOutput)
            mock_run.assert_called_once()

# 2. create_branch calls git commands
@patch("subprocess.run")
def test_create_branch_calls_git(mock_run, tmp_path):
    params = mcp_server_module.CreateBranchInput(repo_path=str(tmp_path), branch_name="feat", base_branch="main")
    # change dir operations will happen; ensure path exists
    os.makedirs(params.repo_path, exist_ok=True)
    res = mcp_server_module.create_branch.fn(params, None)
    assert res.created_branch == "feat"
    assert mock_run.call_count >= 4

# 3. switch_branch
@patch("subprocess.run")
def test_switch_branch(mock_run, tmp_path):
    params = mcp_server_module.SwitchBranchInput(repo_path=str(tmp_path), branch_name="dev")
    os.makedirs(params.repo_path, exist_ok=True)
    res = mcp_server_module.switch_branch.fn(params, None)
    assert res.active_branch == "dev"
    mock_run.assert_called()

# 4. commit_and_push
@patch("subprocess.run")
def test_commit_and_push(mock_run, tmp_path):
    params = mcp_server_module.CommitPushInput(repo_path=str(tmp_path), branch="dev", commit_message="msg")
    os.makedirs(params.repo_path, exist_ok=True)
    res = mcp_server_module.commit_and_push.fn(params, None)
    assert res.pushed is True
    assert res.branch == "dev"

# 5. create_pull_request uses GitHub client - happy path
@patch.object(mcp_server_module.GITHUB_CLIENT, "get_repo")
def test_create_pull_request(mock_get_repo):
    fake_repo = MagicMock()
    fake_pr = MagicMock(number=123, html_url="http://pr")
    fake_repo.create_pull.return_value = fake_pr
    mock_get_repo.return_value = fake_repo
    params = mcp_server_module.PRInput(repo_fullname="org/repo", source_branch="feat", title="t", body="b")
    res = mcp_server_module.create_pull_request.fn(params, None)
    assert res.pr_number == 123
    assert "http" in res.pr_url

# 6. list_branches returns names
@patch.object(mcp_server_module.GITHUB_CLIENT, "get_repo")
def test_list_branches(mock_get_repo):
    fake_repo = MagicMock()
    branch1 = MagicMock(name="main")
    branch2 = MagicMock(name="dev")
    fake_repo.get_branches.return_value = [branch1, branch2]
    mock_get_repo.return_value = fake_repo
    params = mcp_server_module.BranchListInput(repo_fullname="org/repo")
    res = mcp_server_module.list_branches.fn(params, None)
    assert "branches" in res.dict()
    assert "main" in res.branches or "dev" in res.branches
