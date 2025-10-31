import streamlit as st
import asyncio
import json
import os
from dotenv import load_dotenv
from fastmcp import Client
from openai import AsyncAzureOpenAI

load_dotenv()

# Azure OpenAI setup
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# MCP endpoints
MCP_FINOPS_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8100/mcp")
MCP_INFRA_URL = os.getenv("INFRA_MCP_URL", "http://localhost:8200/mcp")

# Initialize LLM client
llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-05-01-preview",
)

# -------------------------------------------------------------
# 1️⃣ Utility: Ask LLM to decide which tool to run
# -------------------------------------------------------------
async def ask_llm(user_input: str) -> dict:
    """Ask LLM to decide which single tool to call and its parameters."""
    system_prompt = """
You are an MCP Orchestrator.

Your job is to decide which SINGLE TOOL to call based on user intent.

You have access to two MCP servers:
1. Infra MCP → Terraform-related tools (e.g. 'analyze_infra', 'update_infra', etc.)
2. FinOps MCP → FinOps/GitHub-related tools (e.g. 'list_tables', 'describe_table', 'get_rightsizing_recommendations', 'clone_repo', etc.)

Return a JSON object describing which tool to call and its parameters.

Examples:

User: "Analyze Terraform repo at path /repos/demo"
Return:
{
  "tool": "analyze_infra",
  "params": {"repo_path": "/repos/demo"},
  "mcp_server": "infra"
}

User: "Show me all DynamoDB tables"
Return:
{
  "tool": "list_tables",
  "params": {},
  "mcp_server": "finops"
}

User: "Get right-size recommendations for dev environment"
Return:
{
  "tool": "get_rightsizing_recommendations",
  "params": {"environment": "dev"},
  "mcp_server": "finops"
}

If the intent is unclear, return:
{
  "tool": "none",
  "message": "Sorry, I didn’t understand that request."
}

Output strictly JSON — no markdown, no commentary.
"""
    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )
    output = response.choices[0].message.content.strip()
    return json.loads(output.replace("```json", "").replace("```", "").strip())


# -------------------------------------------------------------
# 2️⃣ MCP helper
# -------------------------------------------------------------
async def call_mcp_tool(mcp_url: str, tool_name: str, params: dict):
    """Call any MCP server tool and return clean result."""
    async with Client(mcp_url) as client:
        response = await client.call_tool(tool_name, params)
        # Simplify tabular data display
        if isinstance(response, dict) and "columns" in response and "rows" in response:
            import pandas as pd
            df = pd.DataFrame(response["rows"], columns=response["columns"])
            return df.to_markdown(index=False)
        return response


# -------------------------------------------------------------
# 3️⃣ Orchestrate single-tool operations only
# -------------------------------------------------------------
async def orchestrate(user_input: str):
    llm_output = await ask_llm(user_input)
    tool = llm_output.get("tool")

    if tool == "none":
        return llm_output.get("message", "Unrecognized request.")

    params = llm_output.get("params", {})
    mcp_server = llm_output.get("mcp_server", "finops")

    if mcp_server == "infra":
        return await call_mcp_tool(MCP_INFRA_URL, tool, params)
    else:
        return await call_mcp_tool(MCP_FINOPS_URL, tool, params)


# -------------------------------------------------------------
# 4️⃣ Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Tool Orchestrator")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "activity" not in st.session_state:
    st.session_state.activity = []

# Sidebar logs
st.sidebar.header("🪵 MCP Activity Log")
for entry in st.session_state.activity:
    st.sidebar.write(entry)

# Chat UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask your FinOps or Infra question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Calling appropriate MCP tool..."):
            try:
                result = asyncio.run(orchestrate(prompt))
            except Exception as e:
                result = f"❌ Error: {e}"

            st.session_state.activity.append(result)
            st.write(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
