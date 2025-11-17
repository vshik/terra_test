# rightsize_graph.py
import asyncio
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from wrappers.mcp_wrappers import (
    mcp_get_rightsizing,
    mcp_clone_repo,
    mcp_analyze_infra,
    mcp_update_infra
)

# -----------------------------
# 1️. STATE MODEL
# -----------------------------
class RightSizeState(BaseModel):
    environment: str
    app_id: str
    repo_url: str

    recommendations: Optional[List[Dict[str, Any]]] = None
    analysis: Optional[Dict[str, Any]] = None
    updated_infra: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -----------------------------
# 2️. NODES
# -----------------------------
async def fetch_recommendations(state: RightSizeState):
    try:
        result = await mcp_get_rightsizing(
            environment=state.environment,
            app_id=state.app_id
        )
        return {"recommendations": result}
    except Exception as e:
        return {"error": f"fetch_recommendations failed: {e}"}


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


async def update_infra_node(state: RightSizeState):
    try:
        result = await mcp_update_infra(
            repo_url=state.repo_url,
            analysis=state.analysis,
            recommendations=state.recommendations
        )
        return {"updated_infra": result}
    except Exception as e:
        return {"error": f"update_infra failed: {e}"}


# -----------------------------
# 3️. GRAPH DEFINITION
# -----------------------------
def build_rightsize_graph():
    workflow = StateGraph(RightSizeState)

    # Register nodes
    workflow.add_node("fetch_recommendations", fetch_recommendations)
    workflow.add_node("clone_repository", clone_repository)
    workflow.add_node("analyze_infra", analyze_infra_node)
    workflow.add_node("update_infra", update_infra_node)

    workflow.set_entry_point("fetch_recommendations")

    # Graph transitions
    workflow.add_edge("fetch_recommendations", "clone_repository")
    workflow.add_edge("clone_repository", "analyze_infra")
    workflow.add_edge("analyze_infra", "update_infra")
    workflow.add_edge("update_infra", END)

    return workflow.compile()
