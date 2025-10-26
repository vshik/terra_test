import os
import tempfile
import subprocess
import pyodbc
from typing import List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from github import Github
from fastmcp import FastMCP, Context
from fastmcp.transports.http import HTTPTransport

# ==========================================================
# Load environment
# ==========================================================
load_dotenv()

# ==========================================================
# Unified MCP Server
# ==========================================================
mcp = FastMCP("finops_github_mcp")

# ==========================================================
# GitHub Section
# ==========================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Missing GITHUB_TOKEN in environment.")

GITHUB_CLIENT = Github(GITHUB_TOKEN)


# ---------- GitHub Tools ----------

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


# ==========================================================
# FinOps Section (Azure Synapse)
# ==========================================================

SYNAPSE_CONNSTR = os.getenv("SYNAPSE_CONNECT_STRING")
if not SYNAPSE_CONNSTR:
    raise RuntimeError("Missing SYNAPSE_CONNECT_STRING in environment.")

def get_synapse_conn():
    return pyodbc.connect(SYNAPSE_CONNSTR, autocommit=True)


class FinOpsQueryInput(BaseModel):
    resource_type: str
    region: str
    limit: int = 20

class FinOpsQueryOutput(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

@mcp.tool(name="get_rightsizing_recommendations", description="Query Synapse FinOps DB for right-sizing suggestions")
def get_rightsizing_recommendations(params: FinOpsQueryInput, ctx: Context) -> FinOpsQueryOutput:
    conn = get_synapse_conn()
    cur = conn.cursor()

    query = f"""
        SELECT TOP {params.limit}
            resource_type,
            region,
            current_size,
            recommended_size,
            monthly_saving_usd
        FROM dbo.rightsizing
        WHERE resource_type = ? AND region = ?
        ORDER BY monthly_saving_usd DESC
    """
    cur.execute(query, (params.resource_type, params.region))
    rows = [list(r) for r in cur.fetchall()]
    cols = [c[0] for c in cur.description]

    conn.close()
    return FinOpsQueryOutput(columns=cols, rows=rows)


# ==========================================================
# Main 
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("MCP_PORT", 8100))
    print(f"Unified FinOps+GitHub MCP Server (Synapse) running at http://localhost:{PORT}/mcp")
    transport = HTTPTransport(host="0.0.0.0", port=PORT, path="/mcp")
    mcp.run(transport)
