from fastmcp import FastMCP, Context
from pydantic import BaseModel
from tools.github.clone_repo import clone_repo
from tools.finops.get_recommendations import get_rightsizing_recommendations
# ... and other tools ...

mcp = FastMCP(name="OpsFinOps MCP")

class CloneInput(BaseModel):
    repo_url: str
    branch: str = "main"

@mcp.tool()
def clone_repo_tool(params: CloneInput, ctx: Context):
    return clone_repo(params)

# Similarly register commit_and_push, create_pull_request, get_rightsizing_recommendations, etc.

if __name__ == "__main__":
    mcp.run(port=8500)
