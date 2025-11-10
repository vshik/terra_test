# orchestrator_ui.py
import streamlit as st
import asyncio
import json
import os
from orchestrator_core import orchestrate

# --- Configuration ---
SESSION_STATE_FILE = "session_state.json"

# --- Helpers for persistent state ---
def load_state():
    """Load chat messages and MCP logs from JSON file."""
    if os.path.exists(SESSION_STATE_FILE):
        try:
            with open(SESSION_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    data.get("messages", []),
                    data.get("mcp_logs", []),
                )
        except Exception:
            return [], []  # corrupted file → start clean
    return [], []


def save_state(messages, mcp_logs):
    """Persist chat messages + MCP logs to file."""
    try:
        with open(SESSION_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"messages": messages, "mcp_logs": mcp_logs},
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        print(f"⚠️ Failed to save session state: {e}")


# --- Streamlit setup ---
st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator")

# Initialize state from persistent storage
if "messages" not in st.session_state or "mcp_logs" not in st.session_state:
    messages, logs = load_state()
    st.session_state.messages = messages
    st.session_state.mcp_logs = logs

# --- Sidebar: MCP Logs ---
with st.sidebar:
    st.header("📜 MCP Activity Log")
    if st.session_state.mcp_logs:
        for log in reversed(st.session_state.mcp_logs):
            st.markdown(
                f"**Tool:** `{log['tool']}`  \n"
                f"**Params:** `{log['params']}`  \n"
                f"**Status:** {log['status']}"
            )
            st.divider()
    else:
        st.info("No MCP activity yet...")

    # Button to clear everything
    if st.button("🧹 Clear All Data"):
        st.session_state.messages = []
        st.session_state.mcp_logs = []
        save_state([], [])
        st.experimental_rerun()

# --- Chat Interface ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question or request..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Orchestrate and get result
            result = asyncio.run(
                orchestrate(prompt, st.session_state.mcp_logs, st.session_state.messages)
            )

            # Prepare response
            if isinstance(result, dict) and result.get("tool") == "none":
                reply = result.get("message")
            else:
                reply = str(result)

            # Display + store assistant reply
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # ✅ Persist everything
            save_state(st.session_state.messages, st.session_state.mcp_logs)
