# orchestrator_core.py
import asyncio
import json
import os
from dotenv import load_dotenv
from fastmcp import Client
from openai import AsyncAzureOpenAI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8100/mcp")

llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-05-01-preview",
)


async def ask_llm(user_input: str, conversation_history: list) -> str:
    """Ask LLM to decide which tool to call or return a friendly message, with memory."""
    system_prompt = """
You are the Orchestrator for the Unified MCP Server that controls FinOps and GitHub tools.
Keep context across messages.

If the user asks greetings or general chat, respond directly.
If the user asks for FinOps or GitHub tasks, return a JSON as:
  {"tool": "tool_name", "params": {...}}

Otherwise:
  {"tool": "none", "params": {}, "message": "Hi there! I can help with GitHub and FinOps operations."}
"""

    # Keep last 10 turns only
    trimmed_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

    # Prepare message list for LLM
    messages = [{"role": "system", "content": system_prompt}] + trimmed_history + [
        {"role": "user", "content": user_input}
    ]

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def safe_parse_llm_output(output: str):
    """Parse LLM output safely."""
    try:
        cleaned = output.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        try:
            return eval(cleaned)
        except Exception:
            return {"tool": "none", "params": {}, "message": "Sorry, I couldn't parse the LLM response."}


async def call_mcp_tool(tool_name: str, params: dict):
    """Call a tool on the Unified MCP Server."""
    async with Client(MCP_URL) as client:
        response = await client.call_tool(tool_name, params)
        return response


async def orchestrate(user_input: str, mcp_logs: list, conversation_history: list):
    """Core orchestration logic for the chatbot with context."""
    llm_decision = await ask_llm(user_input, conversation_history)
    parsed = safe_parse_llm_output(llm_decision)

    tool = parsed.get("tool")
    params = parsed.get("params", {})
    message = parsed.get("message")

    if tool == "none":
        return {"tool": "none", "message": message or "Hi there! How can I help you today?"}

    if tool:
        result = await call_mcp_tool(tool, params)
        mcp_logs.append({
            "tool": tool,
            "params": params,
            "status": "Success"
        })
        return result

    return {"tool": "none", "message": "I couldn’t determine which tool to use."}
