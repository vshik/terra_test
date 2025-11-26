import streamlit as st
import asyncio
import json
import os
import httpx

# -------------------------------
# Config
# -------------------------------
ORCHESTRATOR_URL = os.getenv("ORCH_URL", "http://localhost:8001/chat")
SESSION_STATE_FILE = "session_state.json"


# -------------------------------
# Local file-based history persistence
# -------------------------------
def load_state():
    """Load chat and MCP logs from disk."""
    if os.path.exists(SESSION_STATE_FILE):
        try:
            with open(SESSION_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", []), data.get("mcp_logs", [])
        except Exception:
            return [], []
    return [], []


def save_state(messages, mcp_logs):
    """Save chat and MCP logs to disk."""
    try:
        with open(SESSION_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"messages": messages, "mcp_logs": mcp_logs},
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        print("Failed to save session:", e)


# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Astra AI System", layout="wide")
st.title("🤖 Astra Orchestrator Chat 🔮")

# Load state if missing
if "messages" not in st.session_state or "mcp_logs" not in st.session_state:
    messages, logs = load_state()
    st.session_state.messages = messages
    st.session_state.mcp_logs = logs


# -------------------------------
# Sidebar: MCP Logs
# -------------------------------
with st.sidebar:
    st.header("MCP Activity Log")
    if st.session_state.mcp_logs:
        for log in reversed(st.session_state.mcp_logs):
            st.markdown(
                f"**Tool:** `{log['tool']}`  \n"
                f"**Params:** `{log['params']}`  \n"
                f"**Status:** `{log['status']}`"
            )
            st.divider()
    else:
        st.info("No MCP activity yet...")


# Clear all data
if st.button("Clear All Data"):
    st.session_state.messages = []
    st.session_state.mcp_logs = []
    save_state([], [])
    st.rerun()


# -------------------------------
# Render chat history
# -------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------------
# Chat input
# -------------------------------
prompt = st.chat_input("Type your command...")

if prompt:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --- Call orchestrator service ---
    async def call_orchestrator(user_msg, history):
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(
                ORCHESTRATOR_URL,
                json={
                    "message": user_msg,
                    "history": history,
                },
            )
        response.raise_for_status()
        return response.json()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                result = asyncio.run(
                    call_orchestrator(prompt, st.session_state.messages)
                )
            except Exception as e:
                st.error(f"Failed to reach Orchestrator: {e}")
                st.stop()

            reply = result.get("reply", "No response")
            new_logs = result.get("mcp_logs", [])

            # Append logs if returned
            if new_logs:
                st.session_state.mcp_logs.extend(new_logs)

            # Show reply
            st.write(reply)

            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Persist to disk
            save_state(st.session_state.messages, st.session_state.mcp_logs)
