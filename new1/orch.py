# orchestrator_api.py
import os
import json
import asyncio
import ast
import re
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from fastmcp import Client
from openai import AsyncAzureOpenAI


# ============================================================
# Load ENV
# ============================================================
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

MCP_PORT = os.getenv("MCP_PORT", "9000")
MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", f"http://localhost:{MCP_PORT}/mcp")


# ============================================================
# FastAPI App Initialization
# ============================================================
app = FastAPI(title="Astra Orchestrator API")

# Allow Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Azure OpenAI client
# ============================================================
llm_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


# ============================================================
# Pydantic Models for API
# ============================================================
class ChatRequest(BaseModel):
    user_input: str
    history: List[Dict[str, str]] = []
    mcp_logs: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    result: Any
    history: List[Dict[str, str]]
    mcp_logs: List[Dict[str, Any]]


# ============================================================
# LLM router
# ============================================================
async def ask_llm(user_input: str, history: List[Dict[str, str]]) -> str:
    system_prompt = """
You are the Orchestrator for the Astra MCP automation system.
Return STRICT JSON only.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_input})

    resp = await llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# Parse LLM output safely
# ============================================================
def safe_parse_llm_output(raw_text: str):
    clean = raw_text.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    matches = re.findall(r"\{.*\}", clean, flags=re.DOTALL)
    if matches:
        clean = matches[-1]

    try:
        return json.loads(clean)
    except:
        pass

    return ast.literal_eval(clean)


# ============================================================
# Call MCP Tool
# ============================================================
async def call_mcp_tool(tool: str, params: dict):
    async with Client(MCP_URL) as client:
        return await client.call_tool(tool, params)


# ============================================================
# Serialize MCP results
# ============================================================
def serialize_result(v: Any):
    if isinstance(v, BaseModel):
        return v.model_dump()
    if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "__dict__"):
        return {k: serialize_result(val) for k, val in v.__dict__.items()}
    return str(v)


# ============================================================
# Main orchestration
# ============================================================
async def orchestrate(user_input: str, messages: List[Dict], mcp_logs: List[Dict]):
    try:
        raw = await ask_llm(user_input, messages)
        llm_output = safe_parse_llm_output(raw)
    except Exception as e:
        return {"error": f"LLM failed: {e}"}

    workflow = llm_output.get("workflow")

    # Greetings
    if workflow == "none":
        msg = llm_output.get("message", "Hello!")
        messages.append({"role": "assistant", "content": msg})
        return msg

    # Special rightsize workflow
    if workflow in ("update_rightsize_workflow", "update_rightsite_workflow"):
        params = llm_output["params"]
        repo = params.get("repo") or params.get("repo_url")
        env = params.get("environment")

        for name in [
            "run_update_rightsize_workflow",
            "update_rightsize_workflow",
            "update_infra_workflow",
        ]:
            try:
                result = await call_mcp_tool(name, {"repo_url": repo, "environment": env})
                mcp_logs.append({"tool": name, "params": params, "status": "Success"})

                messages.append({
                    "role": "assistant",
                    "content": f"Workflow `{name}` completed."
                })

                return serialize_result(result)

            except Exception:
                continue

        return {"error": "No rightsize workflow tool worked."}

    # Single MCP tool
    if workflow == "single_tool":
        tool = llm_output["tool"]
        params = llm_output["params"]

        result = await call_mcp_tool(tool, params)
        mcp_logs.append({"tool": tool, "params": params, "status": "Success"})

        messages.append({
            "role": "assistant",
            "content": f"Tool `{tool}` executed."
        })

        return serialize_result(result)

    # General LLM question
    if workflow == "general_question":
        answer = llm_output["answer"]
        messages.append({"role": "assistant", "content": answer})
        return answer

    return {"error": "unknown workflow", "llm_output": llm_output}


# ============================================================
# FASTAPI ENDPOINT
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Streamlit posts:
        {
            "user_input": "...",
            "history": [...],
            "mcp_logs": [...]
        }
    """
    result = await orchestrate(req.user_input, req.history, req.mcp_logs)

    return ChatResponse(
        result=result,
        history=req.history,
        mcp_logs=req.mcp_logs,
    )
