# rightsize_graph.py
import asyncio
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from wrappers.mcp_wrappers import (
    mcp_get_rightsizing,
    mcp_get_cmdb_metadata,
    mcp_clone_repo,
    mcp_analyze_infra,
    mcp_update_infra,
    mcp_push_and_commit,
    mcp_raise_pr,
)

# -----------------------------------------
# 1️. STATE MODEL
# -----------------------------------------
class RightSizeState(BaseModel):
    environment: str
    app_id: str
    repo_url: str

    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    cmdb_metadata: Optional[Dict[str, Any]] = None
    updated_infra: Optional[Dict[str, Any]] = None

    clone_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -----------------------------------------
# 2️. NODES
# -----------------------------------------
async def clone_repository(state: RightSizeState):
    try:
        result = await mcp_clone_repo(repo_url=state.repo_url)
        return {"clone_result": result}
    except Exception as e:
        return {"error": f"clone_repository failed: {e}"}


async def analyze_infra_node(state: RightSizeState):
    try:
        result = await mcp_analyze_infra(repo_url=state.repo_url)
        return {"analysis": result}
    except Exception as e:
        return {"error": f"analyze_infra failed: {e}"}


async def fetch_recommendations(state: RightSizeState):
    try:
        result = await mcp_get_rightsizing(
            environment=state.environment,
            app_id=state.app_id
        )
        return {"recommendations": result}
    except Exception as e:
        return {"error": f"fetch_recommendations failed: {e}"}


async def get_cmdb_metadata(state: RightSizeState):
    try:
        result = await mcp_get_cmdb_metadata(
            environment=state.environment,
            app_id=state.app_id
        )
        return {"cmdb_metadata": result}
    except Exception as e:
        return {"error": f"get_cmdb_metadata failed: {e}"}


async def update_infra_node(state: RightSizeState):
    try:
        result = await mcp_update_infra(
            repo_url=state.repo_url,
            analysis=state.analysis,
            recommendations=state.recommendations,
            cmdb=state.cmdb_metadata
        )
        return {"updated_infra": result}
    except Exception as e:
        return {"error": f"update_infra failed: {e}"}


async def push_and_commit(state: RightSizeState):
    try:
        return {"commit": await mcp_push_and_commit(repo_url=state.repo_url)}
    except Exception as e:
        return {"error": f"push_and_commit failed: {e}"}


async def raise_pr_node(state: RightSizeState):
    try:
        return {"pr": await mcp_raise_pr(repo_url=state.repo_url)}
    except Exception as e:
        return {"error": f"raise_pr failed: {e}"}


async def notify_stakeholders(state: RightSizeState):
    try:
        await mcp_notify(state)
        return {}
    except Exception as e:
        return {"error": f"notify_stakeholders failed: {e}"}


# -----------------------------------------
# 3️. ROUTER NODE — END EARLY LOGIC
# -----------------------------------------
def check_recommendations(state: RightSizeState):
    """If recommendations == [{}], stop workflow."""
    recos = state.recommendations

    if recos == [{}] or recos == []:
        return END
    return "update_infra"


# -----------------------------------------
# 4️. GRAPH DEFINITION
# -----------------------------------------
def build_rightsize_graph():
    graph = StateGraph(RightSizeState)

    # Register nodes
    graph.add_node("clone_repository", clone_repository)
    graph.add_node("analyze_infra", analyze_infra_node)
    graph.add_node("fetch_recommendations", fetch_recommendations)
    graph.add_node("get_cmdb_metadata", get_cmdb_metadata)
    graph.add_node("check_recommendations", check_recommendations)
    graph.add_node("update_infra", update_infra_node)
    graph.add_node("push_and_commit", push_and_commit)
    graph.add_node("raise_pr", raise_pr_node)
    graph.add_node("notify_stakeholders", notify_stakeholders)

    # Entry point
    graph.set_entry_point("clone_repository")

    # Linear part
    graph.add_edge("clone_repository", "analyze_infra")

    # Parallel branches
    graph.add_edge("analyze_infra", "fetch_recommendations")
    graph.add_edge("analyze_infra", "get_cmdb_metadata")

    # Merge: when both branches finish → check_recommendations
    graph.add_edge("fetch_recommendations", "check_recommendations")
    graph.add_edge("get_cmdb_metadata", "check_recommendations")

    # Router logic
    graph.add_conditional_edges(
        "check_recommendations",
        check_recommendations,
        {
            END: END,
            "update_infra": "update_infra",
        }
    )

    # Final steps
    graph.add_edge("update_infra", "push_and_commit")
    graph.add_edge("push_and_commit", "raise_pr")
    graph.add_edge("raise_pr", END)

    return graph.compile()
