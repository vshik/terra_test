# orchestrator_core.py
import asyncio
import json
import os
from dotenv import load_dotenv

from rightsize_graph import build_rightsize_graph, RightSizeState
from wrappers.mcp_wrappers import (
    mcp_clone_repo,
    mcp_get_rightsizing,
    mcp_analyze_infra,
    mcp_update_infra
)

# ---------------------------
# Load environment + LLM client
# ---------------------------
load_dotenv()
from openai import AsyncAzureOpenAI

llm_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-05-01-preview",
)


# ---------------------------
# LLM DECISION MAKER
# ---------------------------
async def ask_llm(query: str):
    system = """
You decide between:
1. Single-tool:
   - clone_repo
   - analyze_infra
   - get_rightsizing_recommendations
2. Special workflow:
   - update_rightsize_workflow(environment, app_id, repo_url)
Return ONLY JSON.
"""
    resp = await llm_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": query}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def safe_parse(o: str):
    try:
        return json.loads(o)
    except Exception:
        return {"tool": "none", "params": {}, "message": "Invalid JSON from LLM"}


# ---------------------------
# ORCHESTRATOR ROUTER
# ---------------------------
async def orchestrate(user_msg: str, mcp_logs: list):
    decision = safe_parse(await ask_llm(user_msg))
    tool = decision.get("tool")
    params = decision.get("params", {})

    if tool == "none":
        return {"message": decision.get("message", "OK")}

    # 📌 CASE 1 — GRAPH WORKFLOW
    if tool == "update_rightsize_workflow":
        graph = build_rightsize_graph()
        state = RightSizeState(**params)
        result = await graph.invoke(state)

        mcp_logs.append({"tool": tool, "params": params, "status": "OK"})
        return result

    # 📌 CASE 2 — DIRECT MCP TOOL CALL
    if tool == "clone_repo":
        result = await mcp_clone_repo(**params)
    elif tool == "get_rightsizing_recommendations":
        result = await mcp_get_rightsizing(**params)
    elif tool == "analyze_infra":
        result = await mcp_analyze_infra(**params)
    else:
        return {"message": f"Unknown tool {tool}"}

    mcp_logs.append({"tool": tool, "params": params, "status": "OK"})
    return result
