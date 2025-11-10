# orchestrator_ui.py
import streamlit as st
import asyncio
from orchestrator_core import orchestrate

st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mcp_logs" not in st.session_state:
    st.session_state.mcp_logs = []

with st.sidebar:
    st.header("📜 MCP Activity Log")
    if st.session_state.mcp_logs:
        for log in reversed(st.session_state.mcp_logs):
            st.markdown(f"**Tool:** `{log['tool']}`  \n**Params:** `{log['params']}`  \n**Status:** {log['status']}")
            st.divider()
    else:
        st.info("No MCP activity yet...")

# --- Chat Interface ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question or request..."):
    # Add new user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(
                orchestrate(prompt, st.session_state.mcp_logs, st.session_state.messages)
            )

            # Determine assistant message
            if isinstance(result, dict) and result.get("tool") == "none":
                reply = result.get("message")
            else:
                reply = str(result)

            # Display + save message
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
