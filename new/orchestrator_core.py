# orchestrator/orchestrator_core.py
import os
import asyncio
import json
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from fastmcp import Client
from .utils import safe_parse_llm_output, serialize_result
from .workflows.rightsize_graph import RightSizeWorkflowRunner

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8100/mcp")

llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
)

# ----- LLM prompt (router) -----
SYSTEM_PROMPT = """
You are the Orchestrator between user requests and two systems:
1) Single MCP tools (one-shot)
2) The special 'rightsize_workflow'

If the user asks to update right-size variables for a repo, return:
{"workflow": "rightsize", "params": {"repo_url": "...", "environment": "...", "app_id": "...", "branch": "..."}}

If user intends a single tool call (clone_repo, get_rightsizing_recommendations, analyze_infra, update_infra, commit_and_push, create_pull_request, list_branches),
return:
{"workflow": "single_tool", "tool": "clone_repo", "params": {"repo_url": "...", ...}}

If user is greeting or asking a casual question, return:
{"workflow": "none", "message": "friendly text"}

If user is asking for a general knowledge answer, return:
{"workflow": "general_question", "answer": "text response"}

Return strictly JSON (no markdown).
"""

async def ask_llm(user_input: str, history: list = None) -> dict:
    """Ask router LLM to decide the workflow or tool"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # include last 6 turns
        for m in history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})
    resp = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    parsed = safe_parse_llm_output(raw)
    return parsed

# ----- MCP call helper -----
async def call_mcp_tool(tool_name: str, params: dict):
    """Call tool on unified MCP server. Simple wrapper."""
    # fastmcp Client usage: ensure params is a dict with the right shape expected by your MCP tool
    async with Client(MCP_URL) as client:
        # many MCP servers expect (params) directly; adjust if your version expects {"params": {...}}
        try:
            response = await client.call_tool(tool_name, params)
        except TypeError:
            # some fastmcp versions require {"params": params}
            response = await client.call_tool(tool_name, {"params": params})
        return response

# ----- Orchestration entry point -----
async def orchestrate(user_input: str, mcp_logs: list = None, history: list = None):
    mcp_logs = mcp_logs if mcp_logs is not None else []
    history = history if history is not None else []

    parsed = await ask_llm(user_input, history)
    workflow = parsed.get("workflow")
    if workflow == "none":
        return {"tool": "none", "message": parsed.get("message", "Hi! How can I help?")}
    if workflow == "general_question":
        return {"tool": "none", "message": parsed.get("answer")}

    if workflow == "single_tool":
        tool = parsed["tool"]
        params = parsed.get("params", {})
        try:
            result = await call_mcp_tool(tool, params)
            mcp_logs.append({"tool": tool, "params": params, "status": "Success"})
            return result
        except Exception as e:
            mcp_logs.append({"tool": tool, "params": params, "status": f"Error: {e}"})
            return {"error": str(e)}

    if workflow == "rightsize":
        # Gather expected params
        params = parsed.get("params", {})
        repo_url = params.get("repo_url") or params.get("repo")  # allow both
        environment = params.get("environment")
        app_id = params.get("app_id") or params.get("humanaID") or params.get("app")
        branch = params.get("branch", "auto-rightsize")
        base_dir = params.get("base_dir", None)
        analysis_out_dir = params.get("analysis_out_dir", None)
        dry_run = params.get("dry_run", False)

        # instantiate runner
        runner = RightSizeWorkflowRunner(call_mcp_tool)
        try:
            result = await runner.run(repo_url=repo_url,
                                      environment=environment,
                                      app_id=app_id,
                                      branch=branch,
                                      base_dir=base_dir,
                                      analysis_out_dir=analysis_out_dir,
                                      dry_run=dry_run)
            # Log result
            status = result.get("status", "unknown")
            mcp_logs.append({"tool": "rightsize_workflow", "params": params, "status": status})
            return result
        except Exception as e:
            mcp_logs.append({"tool": "rightsize_workflow", "params": params, "status": f"Error: {e}"})
            return {"error": str(e)}

    return {"error": "Unknown workflow type."}
