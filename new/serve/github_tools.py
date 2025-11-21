# github_tools.py
import os
import tempfile
import subprocess
from typing import Optional, List
from pydantic import BaseModel
from github import Github

from fastmcp import FastMCP, Context

# ----------------------------------------
# GitHub initialization
# ----------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN in environment.")

GITHUB_CLIENT = Github(GITHUB_TOKEN)

# Exported MCP instance
mcp = FastMCP("github_tools")

# ----------------------------------------
# GitHub Tools
# ----------------------------------------

class CloneInput(BaseModel):
    repo_url: str
    branch: Optional[str] = "main"

class CloneOutput(BaseModel):
    local_path: str


@mcp.tool(name="clone_repo", description="Clone a GitHub repo to a local temp directory")
def clone_repo(params: CloneInput, ctx: Context) -> CloneOutput:
    tmpdir = tempfile.mkdtemp(prefix="repo_")
    subprocess.run(["git", "clone", "-b", params.branch, params.repo_url, tmpdir], check=True)
    return CloneOutput(local_path=tmpdir)


class CreateBranchInput(BaseModel):
    repo_path: str
    branch_name: str
    base_branch: Optional[str] = "main"

class CreateBranchOutput(BaseModel):
    created_branch: str


@mcp.tool(name="create_branch", description="Create a new Git branch from a base branch")
def create_branch(params: CreateBranchInput, ctx: Context) -> CreateBranchOutput:
    os.chdir(params.repo_path)
    subprocess.run(["git", "fetch", "origin", params.base_branch], check=True)
    subprocess.run(["git", "checkout", params.base_branch], check=True)
    subprocess.run(["git", "pull", "origin", params.base_branch], check=True)
    subprocess.run(["git", "checkout", "-b", params.branch_name], check=True)
    return CreateBranchOutput(created_branch=params.branch_name)


class SwitchBranchInput(BaseModel):
    repo_path: str
    branch_name: str

class SwitchBranchOutput(BaseModel):
    active_branch: str


@mcp.tool(name="switch_branch", description="Switch to an existing local Git branch")
def switch_branch(params: SwitchBranchInput, ctx: Context) -> SwitchBranchOutput:
    os.chdir(params.repo_path)
    subprocess.run(["git", "checkout", params.branch_name], check=True)
    return SwitchBranchOutput(active_branch=params.branch_name)


class CommitPushInput(BaseModel):
    repo_path: str
    branch: str
    commit_message: str = "Automated commit by MCP"

class CommitPushOutput(BaseModel):
    pushed: bool
    branch: str


@mcp.tool(name="commit_and_push", description="Commit local changes and push to GitHub branch")
def commit_and_push(params: CommitPushInput, ctx: Context) -> CommitPushOutput:
    os.chdir(params.repo_path)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", params.commit_message], check=True)
    subprocess.run(["git", "push", "--set-upstream", "origin", params.branch], check=True)
    return CommitPushOutput(pushed=True, branch=params.branch)


class PRInput(BaseModel):
    repo_fullname: str
    source_branch: str
    target_branch: str = "main"
    title: str
    body: Optional[str] = None

class PROutput(BaseModel):
    pr_number: int
    pr_url: str


@mcp.tool(name="create_pull_request", description="Create a pull request in GitHub repo")
def create_pull_request(params: PRInput, ctx: Context) -> PROutput:
    repo = GITHUB_CLIENT.get_repo(params.repo_fullname)
    pr = repo.create_pull(
        title=params.title,
        body=params.body or "",
        head=params.source_branch,
        base=params.target_branch,
    )
    return PROutput(pr_number=pr.number, pr_url=pr.html_url)


class BranchListInput(BaseModel):
    repo_fullname: str

class BranchListOutput(BaseModel):
    branches: List[str]


@mcp.tool(name="list_branches", description="List branches in a GitHub repository")
def list_branches(params: BranchListInput, ctx: Context) -> BranchListOutput:
    repo = GITHUB_CLIENT.get_repo(params.repo_fullname)
    branches = [b.name for b in repo.get_branches()]
    return BranchListOutput(branches=branches)
