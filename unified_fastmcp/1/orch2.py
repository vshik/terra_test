import os
import asyncio
import streamlit as st
from openai import AsyncAzureOpenAI
from fastmcp import Client
from dotenv import load_dotenv

# ──────────────────────────────
# Load environment variables
# ──────────────────────────────
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
FINOPS_GITHUB_MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8500/mcp")

# ──────────────────────────────
# Async Azure OpenAI client
# ──────────────────────────────
llm_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ──────────────────────────────
# Streamlit page setup
# ──────────────────────────────
st.set_page_config(page_title="AI Orchestrator Chat", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 AI Orchestrator — FinOps + GitHub MCP Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []

# Sidebar for tool logs
st.sidebar.header("🧩 MCP Tools Activity Log")
for tool_name, result in st.session_state.tool_log:
    st.sidebar.write(f"**{tool_name}** → {str(result)[:100]}...")

st.sidebar.markdown("---")
st.sidebar.info("This chat orchestrates FinOps (Synapse) and GitHub operations via MCP tools.")

# ──────────────────────────────
# Async MCP tool call
# ──────────────────────────────
async def call_mcp_tool(tool_name: str, params: dict):
    """Call a unified MCP tool asynchronously."""
    async with Client(FINOPS_GITHUB_MCP_URL) as client:
        result = await client.call_tool(tool_name, params)
        st.session_state.tool_log.append((tool_name, result))
        return result

# ──────────────────────────────
# Async LLM interaction
# ──────────────────────────────
async def ask_llm(prompt: str):
    """Ask Azure OpenAI model asynchronously."""
    completion = await llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": (
                "You are an AI Orchestrator assistant that can call tools "
                "from the FinOps + GitHub MCP. You decide which tool to call "
                "based on user intent. Available tools include:\n"
                "- get_recommendations (FinOps)\n"
                "- exec_stored_procedure (FinOps)\n"
                "- run_query (FinOps)\n"
                "- list_tables (FinOps)\n"
                "- create_pull_request, commit_and_push, clone_repo, create_branch, switch_branch (GitHub)\n"
            )},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

# ──────────────────────────────
# Async Orchestration logic
# ──────────────────────────────
async def orchestrate(user_input: str):
    """Determine which MCP tool to call based on user input."""
    response_text = ""
    try:
        # Interpret user query
        llm_decision = await ask_llm(f"User asked: {user_input}\n\nDecide which tool and parameters to call. Respond with JSON format: {{'tool': 'tool_name', 'params': {{...}}}}")
        
        # Try to extract tool + params
        import json
        parsed = None
        try:
            parsed = json.loads(llm_decision.replace("```json", "").replace("```", ""))
        except Exception:
            response_text = f"I couldn’t parse the LLM response properly:\n\n{llm_decision}"
            return response_text

        tool_name = parsed.get("tool")
        params = parsed.get("params", {})

        if not tool_name:
            response_text = f"LLM didn't specify a tool. Raw response:\n\n{llm_decision}"
            return response_text

        # Call MCP tool
        tool_result = await call_mcp_tool(tool_name, params)
        response_text = f" Tool `{tool_name}` executed successfully.\n\nResult:\n{tool_result}"

    except Exception as e:
        response_text = f" Error during orchestration: {str(e)}"

    return response_text

# ──────────────────────────────
# Chat Interface
# ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me to run GitHub or FinOps actions..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response asynchronously
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... ")

        async def handle_async_chat():
            result = await orchestrate(user_input)
            message_placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        asyncio.run(handle_async_chat())
