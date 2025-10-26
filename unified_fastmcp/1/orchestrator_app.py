# orchestrator/app.py
import os
import json
import asyncio
import streamlit as st
import requests
from dotenv import load_dotenv
from fastmcp import AsyncClient

load_dotenv()

# ============
# Config
# ============
# Azure OpenAI (Responses API) settings
AZURE_OPENAI_BASE = os.getenv("AZURE_OPENAI_BASE")  # e.g. https://<resource>.openai.azure.com
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g. "gpt-4.1-deploy"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01")  # adjust if your subscription uses different version

# MCP endpoint (the unified finops+github MCP server)
MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8100/mcp")

# Basic validation
if not AZURE_OPENAI_BASE or not AZURE_DEPLOYMENT or not AZURE_API_KEY:
    st.error("Missing Azure OpenAI config. Set AZURE_OPENAI_BASE, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY in .env")
    st.stop()

# ============
# Helper: call Azure Responses API
#   We will ask the model to return a JSON {tool: <tool_name>, params: {...}}
# ============
def call_azure_openai(prompt: str, system: str = None, max_tokens: int = 512):
    url = f"{AZURE_OPENAI_BASE}/openai/deployments/{AZURE_DEPLOYMENT}/responses?api-version={AZURE_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    payload = {
        "input": prompt,
        "messages": [
            {"role": "system", "content": system or "You are an assistant that returns a single JSON action describing which MCP tool to call and with what params."},
            {"role": "user", "content": prompt}
        ],
        # keep it direct; may be fine to use temperature=0 for deterministic outputs
        "temperature": 0.0,
        "max_output_tokens": max_tokens
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()

    # The Responses API returns structured outputs. The field names may vary with api-version.
    # Attempt to extract the top text output.
    # For newer Responses API versions you may get: resp["output"][0]["content"][0]["text"]
    # Fallback to stringifying.
    text = None
    try:
        # Try common location used by Responses API
        outs = resp.get("output", [])
        if outs and isinstance(outs, list):
            # find first text content
            for item in outs:
                contents = item.get("content", [])
                for c in contents:
                    if c.get("type") in (None, "output_text"):
                        text = c.get("text") or c.get("value") or text
                        if text:
                            break
                if text:
                    break
    except Exception:
        text = None

    if text is None:
        # Fallback - return full response as string
        text = json.dumps(resp)

    return text, resp

# ============
# Helper: call MCP tool (async wrapper)
# ============
async def call_mcp_tool(tool_name: str, params: dict):
    async with AsyncClient(MCP_URL) as client:
        return await client.call_tool(tool_name, params)

# ============
# Streamlit UI
# ============
st.set_page_config(page_title="Orchestrator Chatbot", layout="wide")
st.title("Orchestrator Chatbot — FinOps + GitHub MCP")

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar controls
st.sidebar.header("Settings")
st.sidebar.text("Azure Deployment:")
st.sidebar.write(f"{AZURE_DEPLOYMENT}")
st.sidebar.text("MCP endpoint:")
st.sidebar.write(f"{MCP_URL}")

# Input
user_input = st.text_area("User message (ask the bot to do an action)", height=120, key="user_input")
if st.button("Send"):
    if not user_input.strip():
        st.warning("Enter a message")
    else:
        st.session_state.history.append({"role": "user", "text": user_input})
        # Compose prompt instructing the model to return JSON with a specific schema
        prompt = f"""
You are an agent controller. The user request:

\"\"\"{user_input} \"\"\"

Respond ONLY with a JSON object (no additional text) with this schema:
{{
  "tool": "<mcp_tool_name>",
  "params": {{ ... arbitrary params ... }},
  "explain": "<short textual explanation of what you'll do>"
}}

Available MCP tools (examples): clone_repo, create_branch, switch_branch, commit_and_push, create_pull_request, list_branches, get_rightsizing_recommendations

- Validate fields: tool must be one of the available tools.
- params must be a JSON object appropriate for the tool.
- Keep the JSON minimal and valid.
- Do not include code blocks or markdown. Only output JSON.
"""
        with st.spinner("Calling Azure OpenAI to decide action..."):
            try:
                text, raw = call_azure_openai(prompt)
            except Exception as e:
                st.error(f"Azure OpenAI call failed: {e}")
                st.stop()

        # show the model raw decision (for transparency)
        st.subheader("Model decision (raw)")
        st.code(text)

        # Try to parse JSON
        try:
            action = json.loads(text)
        except Exception as e:
            st.error(f"Could not parse JSON from model output: {e}")
            st.session_state.history.append({"role": "assistant", "text": f"Invalid JSON returned: {e}"})
            st.stop()

        # Simple validation
        tool = action.get("tool")
        params = action.get("params", {})
        explain = action.get("explain", "")

        if not tool:
            st.error("Model did not return a 'tool' field")
            st.stop()

        # Confirm UI: show chosen tool and params, allow user to cancel
        st.write("**Planned action**")
        st.json({"tool": tool, "params": params, "explain": explain})
        if st.button("Execute this action"):
            # Execute via MCP
            with st.spinner(f"Calling MCP tool `{tool}`..."):
                try:
                    result = asyncio.run(call_mcp_tool(tool, params))
                except Exception as e:
                    st.error(f"Failed to call MCP tool: {e}")
                    st.session_state.history.append({"role": "assistant", "text": f"Tool invocation failed: {e}"})
                else:
                    st.success("Tool executed — result below")
                    st.json(result)
                    # Optionally ask the model to produce a user-friendly summary
                    summary_prompt = f"""
I executed the following MCP tool call:

Tool: {tool}
Params: {json.dumps(params)}
Result: {json.dumps(result)}

Provide a short summary for the user (1-3 sentences).
"""
                    summary_text, _ = call_azure_openai(summary_prompt, system="You are a concise assistant that summarizes results.")
                    st.write("**Model summary of result**")
                    st.write(summary_text)
                    st.session_state.history.append({"role": "assistant", "text": summary_text})

# Display history
st.sidebar.header("Conversation history")
for item in st.session_state.history[-20:]:
    who = "User" if item["role"] == "user" else "Assistant"
    st.sidebar.markdown(f"**{who}:** {item['text'][:200]}")
