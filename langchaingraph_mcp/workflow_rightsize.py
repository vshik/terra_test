# orchestrator/workflow_rightsize.py
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import StructuredTool
from orchestrator.utils import call_mcp_tool
import asyncio
import os

# --- MCP URLs ---
MCP_INFRA_URL = os.getenv("INFRA_MCP_URL", "http://localhost:8200/mcp")
MCP_OPS_FINOPS_URL = os.getenv("OPS_FINOPS_MCP_URL", "http://localhost:8100/mcp")


# ---------- Wrap each MCP call as LangChain Tools ----------
clone_tool = StructuredTool.from_function(
    func=lambda repo_url, branch="main": asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "clone_repo", {"repo_url": repo_url, "branch": branch})
    ),
    name="clone_repo",
    description="Clone GitHub repository"
)

finops_tool = StructuredTool.from_function(
    func=lambda environment, humanaID: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "get_rightsizing_recommendations",
                      {"environment": environment, "humanaID": humanaID})
    ),
    name="get_rightsizing_recommendations",
    description="Fetch right-sizing recommendations from FinOps DB"
)

analyze_tool = StructuredTool.from_function(
    func=lambda repo_path: asyncio.run(
        call_mcp_tool(MCP_INFRA_URL, "analyze_tf_file", {"repo_path": repo_path})
    ),
    name="analyze_tf_file",
    description="Analyze Terraform files and identify right-size variables"
)

update_tool = StructuredTool.from_function(
    func=lambda repo_path, analysis_json, recommendations: asyncio.run(
        call_mcp_tool(MCP_INFRA_URL, "update_tf_file", {
            "repo_path": repo_path,
            "analysis_json": analysis_json,
            "recommendations": recommendations
        })
    ),
    name="update_tf_file",
    description="Update Terraform files using FinOps recommendations"
)

commit_tool = StructuredTool.from_function(
    func=lambda repo_path, branch, message: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "commit_and_push", {
            "repo_path": repo_path,
            "branch": branch,
            "commit_message": message
        })
    ),
    name="commit_and_push",
    description="Commit and push updates"
)

pr_tool = StructuredTool.from_function(
    func=lambda repo_url, source_branch, title, body: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "create_pull_request", {
            "repo_url": repo_url,
            "source_branch": source_branch,
            "title": title,
            "body": body
        })
    ),
    name="create_pull_request",
    description="Create pull request"
)

cleanup_tool = StructuredTool.from_function(
    func=lambda local_path: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "cleanup_clone", {"local_path": local_path})
    ),
    name="cleanup_clone",
    description="Cleanup temporary cloned repo"
)


# ---------- Build the LangGraph ----------
def build_rightsize_workflow():
    graph = StateGraph()

    graph.add_node("clone_repo", ToolNode(tool=clone_tool))
    graph.add_node("finops", ToolNode(tool=finops_tool))
    graph.add_node("analyze", ToolNode(tool=analyze_tool))
    graph.add_node("update", ToolNode(tool=update_tool))
    graph.add_node("commit", ToolNode(tool=commit_tool))
    graph.add_node("create_pr", ToolNode(tool=pr_tool))
    graph.add_node("cleanup", ToolNode(tool=cleanup_tool))

    # --- Parallel: clone_repo + finops ---
    graph.add_edge("clone_repo", "analyze")
    graph.add_edge("finops", "analyze")

    # --- Sequential chain ---
    graph.add_edge("analyze", "update")
    graph.add_edge("update", "commit")
    graph.add_edge("commit", "create_pr")
    graph.add_edge("create_pr", "cleanup")

    # --- Error handling: cleanup on failure ---
    def on_error(e, state):
        state["error"] = str(e)
        return "cleanup"

    graph.on_error(on_error)
    return graph.compile()
