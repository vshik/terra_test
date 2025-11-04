import asyncio
import os
import pytest
from unittest.mock import AsyncMock

# Provide async event loop for pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Common monkeypatch for streamlit session where app references it
import streamlit as st
@pytest.fixture(autouse=True)
def clear_streamlit_session(monkeypatch):
    # Ensure session_state has expected keys used by app
    if not hasattr(st, "session_state"):
        monkeypatch.setattr(st, "session_state", {})
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("mcp_logs", [])
    yield
    # cleanup
    st.session_state["messages"].clear()
    st.session_state["mcp_logs"].clear()

# Convenience: set env flag to allow integration tests
@pytest.fixture(scope="session")
def run_integration():
    return os.getenv("RUN_MCP_INTEGRATION", "false").lower() == "true"
