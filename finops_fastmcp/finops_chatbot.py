import os
import asyncio
import streamlit as st
# from fastmcp import FastMCP
from fastmcp.client import MCPClient
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastmcp.openai import get_openai_tools

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MCP_URL = os.getenv("FINOPS_MCP_URL", "http://localhost:8000/mcp")

SYSTEM_PROMPT = """
You are a FinOps Analyst Assistant helping users analyze and optimize cloud costs
using data from an MCP server that exposes controlled tools for FinOps operations.

You have access to the following MCP tools:
- list_tables: to discover available database tables
- describe_table: to understand table schemas and columns
- run_query_safe: to run parameterized SELECT queries safely
- run_aggregate_query: to summarize or group data efficiently
- run_stored_procedure: to execute predefined analytics or optimization routines

Each tool has strict input validation rules. Always send correctly structured JSON
arguments and never attempt to use undefined or unregistered tools.

Your goals are to:
1. Retrieve, summarize, and interpret FinOps metrics using the available MCP tools.
2. Prefer aggregated or summarized outputs over full raw datasets.
3. Validate user intent and clarify parameters such as date ranges, columns, or regions
   before running large queries.
4. Never generate raw SQL or invent new tools, columns, or stored procedures.

If a user asks for something that **you cannot perform with the current MCP tools or stored procedures**, follow this graceful fallback policy:

- Politely explain that no existing tool supports that specific query.
- Offer related or approximate insights using the available tools.
- Suggest alternative phrasing such as “Would you like me to show overall cost trends instead?”
- Never hallucinate data or fabricate tool names.
- Optionally, log or report the missing capability using the `log_event` MCP service (if available).

Example behavior:
User: “Show me Kubernetes pod-level efficiency by namespace.”
You: “I don’t currently have a stored procedure for Kubernetes efficiency.
However, I can retrieve total cloud cost by resource group or application tag.
Would you like me to do that instead?”

You act as a responsible, safety-focused FinOps assistant for enterprise-grade analytics.
Always maintain professionalism, transparency, and data governance integrity.
"""

# Initialize OpenAI and MCP
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
mcp = MCPClient(MCP_URL)

st.set_page_config(page_title="FinOps Chatbot", page_icon="💰", layout="wide")
st.title("FinOps Analyst Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- 🔄 Async query handler ---
async def handle_query(user_message: str):
    """Send user message to GPT-4.1 with available MCP tools."""
    st.session_state.messages.append({"role": "user", "content": user_message})

    # ✅ Dynamically generate OpenAI-compatible MCP tool schemas
    tools = await get_openai_tools(mcp)

    # Call the model
    response = await openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto",
    )

    # Extract assistant reply
    message = response.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": message.content})

    # Render in Streamlit
    with st.chat_message("assistant"):
        st.markdown(message.content or "⚠️ (No reply from assistant)")

    return message


# --- 💬 Chat UI interaction ---
user_input = st.chat_input("Ask a FinOps question (e.g. show top 5 costly resources)")

# Display chat history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# When user sends a message
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    asyncio.run(handle_query(user_input))
