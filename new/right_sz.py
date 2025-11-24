# rightsize_graph.py
import asyncio
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from wrappers.mcp_wrappers import (
    mcp_clone_repo,
    mcp_analyze_infra,
    mcp_get_rightsizing,
    mcp_get_cmdb_metadata,
    mcp_update_infra,
    mcp_push_and_commit,
    mcp_raise_pr,
    mcp_notify,
)

# ==========================================================
# 1️⃣ STATE MODEL
# ==========================================================
class RightSizeState(BaseModel):
    environment: str
    app_id: str
    repo_url: str

    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    cmdb_metadata: Optional[Dict[str, Any]] = None
    updated_infra: Optional[Dict[str, Any]] = None

    clone_result: Optional[Dict[str, Any]] = None
    commit_success: Optional[bool] = None
    route: Optional[str] = None
    error: Optional[str] = None


# ==========================================================
# 2️⃣ WORKFLOW NODES
# ==========================================================

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
        recos = await mcp_get_rightsizing(
            environment=state.environment,
            app_id=state.app_id
        )
        return {"recommendations": recos}
    except Exception as e:
        return {"error": f"fetch_recommendations failed: {e}"}

async def get_cmdb_metadata(state: RightSizeState):
    try:
        meta = await mcp_get_cmdb_metadata(
            environment=state.environment,
            app_id=state.app_id
        )
        return {"cmdb_metadata": meta}
    except Exception as e:
        return {"error": f"get_cmdb_metadata failed: {e}"}

def check_recommendations(state: RightSizeState):
    """Router:
       → END if no actionable recommendations
       → update_infra otherwise
    """
    recos = state.recommendations

    if recos == [{}] or recos == [] or recos is None:
        return {"route": "end"}

    return {"route": "update"}

async def update_infra_node(state: RightSizeState):
    try:
        updated = await mcp_update_infra(
            repo_url=state.repo_url,
            analysis=state.analysis,
            recommendations=state.recommendations,
            cmdb=state.cmdb_metadata,
        )
        return {"updated_infra": updated}
    except Exception as e:
        return {"error": f"update_infra failed: {e}"}

async def push_and_commit(state: RightSizeState):
    try:
        ok = await mcp_push_and_commit(repo_url=state.repo_url)
        return {"commit_success": ok}
    except Exception as e:
        return {"error": f"push_and_commit failed: {e}"}

def route_after_commit(state: RightSizeState):
    """Route:
       → END if commit failed
       → raise_pr if commit succeeded
    """
    if not state.commit_success:
        return {"route": "end"}

    return {"route": "raise_pr"}

async def raise_pr_node(state: RightSizeState):
    try:
        pr = await mcp_raise_pr(repo_url=state.repo_url)
        return {"pr": pr}
    except Exception as e:
        return {"error": f"raise_pr failed: {e}"}

async def notify_stakeholders(state: RightSizeState):
    try:
        await mcp_notify(state)
        return {}
    except Exception as e:
        return {"error": f"notify_stakeholders failed: {e}"}


# ==========================================================
# 3️⃣ BUILD GRAPH
# ==========================================================
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
    graph.add_node("route_after_commit", route_after_commit)
    graph.add_node("raise_pr", raise_pr_node)
    graph.add_node("notify_stakeholders", notify_stakeholders)

    # Entry
    graph.set_entry_point("clone_repository")

    # Linear steps
    graph.add_edge("clone_repository", "analyze_infra")

    # Parallel: recommendations + cmdb metadata
    graph.add_edge("analyze_infra", "fetch_recommendations")
    graph.add_edge("analyze_infra", "get_cmdb_metadata")

    # Merge → router
    graph.add_edge("fetch_recommendations", "check_recommendations")
    graph.add_edge("get_cmdb_metadata", "check_recommendations")

    # Conditional routing based on recos
    graph.add_conditional_edges(
        "check_recommendations",
        lambda s: s.route,
        {
            "end": END,
            "update": "update_infra",
        }
    )

    # Continue flow
    graph.add_edge("update_infra", "push_and_commit")

    # Conditional: PR only if commit succeeded
    graph.add_edge("push_and_commit", "route_after_commit")

    graph.add_conditional_edges(
        "route_after_commit",
        lambda s: s.route,
        {
            "end": END,
            "raise_pr": "raise_pr",
        }
    )

    graph.add_edge("raise_pr", "notify_stakeholders")
    graph.add_edge("notify_stakeholders", END)

    return graph.compile()
