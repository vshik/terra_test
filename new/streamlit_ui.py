# orchestrator/orchestrator_ui.py
import streamlit as st
import asyncio
import json
import os
from .orchestrator_core import orchestrate

STATE_FILE = "orch_session_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", []), data.get("mcp_logs", [])
        except Exception:
            return [], []
    return [], []

def save_state(messages, mcp_logs):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "mcp_logs": mcp_logs}, f, indent=2, ensure_ascii=False)

st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator (Hybrid)")

if "messages" not in st.session_state or "mcp_logs" not in st.session_state:
    messages, logs = load_state()
    st.session_state.messages = messages
    st.session_state.mcp_logs = logs

# Sidebar
with st.sidebar:
    st.header("🪵 MCP Activity Log")
    if st.session_state.mcp_logs:
        for log in reversed(st.session_state.mcp_logs[-100:]):
            st.markdown(f"**Tool:** `{log.get('tool')}`  \n**Params:** `{log.get('params')}`  \n**Status:** `{log.get('status')}`")
            st.divider()
    else:
        st.info("No MCP activity yet...")

# Chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Type your question or request...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # run orchestrate in event loop
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(orchestrate(prompt, st.session_state.mcp_logs, st.session_state.messages))
            except Exception as e:
                result = {"error": str(e)}
            # Pretty display results
            if isinstance(result, dict) and result.get("tool") == "none":
                reply = result.get("message")
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                # For workflow results (dict) we show summary & not dump raw structured output
                if isinstance(result, dict) and "status" in result:
                    st.write(f"Workflow status: **{result.get('status')}**")
                    # show trace
                    trace = result.get("trace", [])
                    for t in trace[-10:]:
                        st.write("• " + str(t))
                    summary_text = json.dumps(result.get("state", {}), default=str, indent=2)
                    st.expander("Workflow state (debug)", expanded=False).write(summary_text)
                    st.session_state.messages.append({"role": "assistant", "content": f"Workflow ran: {result.get('status')}"})
                else:
                    # For single-tool results, show them directly
                    st.write(result)
                    st.session_state.messages.append({"role": "assistant", "content": str(result)})
    # persist
    save_state(st.session_state.messages, st.session_state.mcp_logs)
