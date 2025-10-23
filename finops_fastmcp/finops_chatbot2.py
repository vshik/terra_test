import streamlit as st
import asyncio
import json
from openai import AsyncOpenAI
from fastmcp import Client as MCPClient  # FastMCP v2.x

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MCP_SERVER_URL = "http://127.0.0.1:8000"  # FastMCP server URL
MODEL_NAME = "gpt-4.1"

# OpenAI API key check
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

llm_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# MCP tools schema
MCP_TOOLS = {
    "list_tables": {"params": {"schema": "Optional[str]"}},
    "describe_table": {"params": {"table_name": "str"}},
    "run_query_safe": {"params": {"sql": "str", "parameters": "Optional[List[Any]]", "limit": "int"}},
    "run_aggregate_query": {"params": {"sql": "str"}},
    "run_stored_procedure": {"params": {"procedure_name": "str", "parameters": "Optional[List[Any]]"}},
}

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="💬 MCP-Aware FinOps Chatbot", layout="centered")
st.title("💬 MCP-Aware FinOps Chatbot")
st.caption("GPT-4.1 + FastMCP with auto MCP tool selection")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": f"""
You are a FinOps assistant. You have access to the following MCP tools:

{json.dumps(MCP_TOOLS, indent=2)}

Your task:
1️⃣ Decide if a user query should call an MCP tool.
2️⃣ If yes, output ONLY a JSON object: {{ "tool": "<tool_name>", "params": {{ ... }} }}
3️⃣ If no tool is needed, output: {{ "tool": "none" }}
4️⃣ Do not execute SQL or perform any actions yourself.
5️⃣ Always validate parameters according to the tool schema.
"""}
    ]

# Display previous messages
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------
# Helper: send to LLM
# -------------------------------------------------------------
async def send_to_llm(messages):
    response = await llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

# -------------------------------------------------------------
# Helper: call MCP tool
# -------------------------------------------------------------
async def call_mcp_tool(tool_name: str, params: dict):
    async with MCPClient(MCP_SERVER_URL) as mcp:
        try:
            return await mcp.call_tool(tool_name, params)
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"

# -------------------------------------------------------------
# Async handler
# -------------------------------------------------------------
async def handle_query(user_message: str):
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Step 1: Ask GPT to decide which tool to call
    tool_decision_prompt = [
        {"role": "system", "content": "Decide which MCP tool to call. Output JSON as described."},
        {"role": "user", "content": user_message}
    ]
    tool_decision_response = await send_to_llm(tool_decision_prompt)

    # Step 2: Parse GPT JSON response
    try:
        tool_info = json.loads(tool_decision_response)
    except json.JSONDecodeError:
        tool_info = {"tool": "none"}

    # Step 3: Call MCP tool if GPT chose one
    if tool_info.get("tool") and tool_info["tool"] != "none":
        mcp_result = await call_mcp_tool(tool_info["tool"], tool_info.get("params", {}))
        assistant_text = f"MCP tool '{tool_info['tool']}' returned:\n{mcp_result}"
    else:
        # fallback: let LLM answer directly
        assistant_text = await send_to_llm(st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

# -------------------------------------------------------------
# Streamlit chat input
# -------------------------------------------------------------
user_input = st.chat_input("Ask a FinOps question (e.g., 'list tables' or 'describe table dbo.MyTable')")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    asyncio.run(handle_query(user_input))
