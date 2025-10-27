# orchestrator/app.py
import streamlit as st
import asyncio
from orchestrator.workflow_rightsize import build_rightsize_workflow

st.set_page_config(page_title="Right-Sizing Workflow", layout="wide")
st.title("🤖 Unified MCP Orchestrator (LangGraph based)")

if "logs" not in st.session_state:
    st.session_state.logs = []

workflow = build_rightsize_workflow()


async def run_workflow(repo_url, environment, humanaID):
    """Run the Right-Sizing workflow end-to-end."""
    st.session_state.logs.append(" Starting right-sizing workflow...")
    yield "Starting workflow..."

    try:
        async for event in workflow.astream({
            "repo_url": repo_url,
            "environment": environment,
            "humanaID": humanaID,
            "branch": "main",
            "commit_message": "Auto right-sizing updates",
            "title": f"Right-sizing update for {environment}",
            "body": "Automated FinOps-driven right-sizing PR"
        }):
            node = event.get("node")
            result = event.get("output")
            msg = f" {node} completed" if result else f" Running {node}..."
            st.session_state.logs.append(msg)
            yield msg
    except Exception as e:
        st.session_state.logs.append(f" Error: {str(e)}")
        yield f"Error: {str(e)}"


# --- Streamlit UI ---
with st.sidebar:
    st.header("📋 Workflow Logs")
    log_placeholder = st.empty()
    log_placeholder.write("\n".join(st.session_state.logs))

repo_url = st.text_input("GitHub Repo URL", "https://github.com/example/repo_123.git")
environment = st.text_input("Environment", "npe")
humanaID = st.text_input("Humana ID", "APP123")

if st.button("Run Right-Sizing Workflow"):
    st.session_state.logs.clear()
    st.write("### Running Workflow...")
    progress = st.empty()

    async def runner():
        async for update in run_workflow(repo_url, environment, humanaID):
            progress.write(update)
            with st.sidebar:
                log_placeholder.write("\n".join(st.session_state.logs))

    asyncio.run(runner())
