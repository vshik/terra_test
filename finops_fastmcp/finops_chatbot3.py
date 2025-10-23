import os
import json
import streamlit as st
import asyncio
from openai import AsyncOpenAI
from aiohttp import ClientSession

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1"
MCP_SERVER_URL = "http://127.0.0.1:8000"

# Define available MCP tools
MCP_TOOLS = {
    "describe_table": {"description": "Describe a table in FinOps dataset", "params": ["table_name"]},
    "list_tables": {"description": "List all available tables", "params": []},
    "query_table": {"description": "Run SQL-like query on a table", "params": ["query"]}
}

# ----------------------------------------------------------------------
# Setup LLM client
# ----------------------------------------------------------------------
llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ----------------------------------------------------------------------
# Utility: Enforced JSON LLM call with retry + correction
# ----------------------------------------------------------------------
async def send_to_llm(messages, expect_json=False, max_retries=2):
    """
    Send messages to GPT model.
    If expect_json=True, enforce JSON mode and retry on invalid output.
    """
    for attempt in range(max_retries):
        try:
            if expect_json:
                response = await llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            else:
                response = await llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )

            content = response.choices[0].message.content.strip()
            # Try to parse JSON to verify it's valid
            json.loads(content)
            return content

        except Exception as e:
            if attempt < max_retries - 1:
                # Add a clarifying correction message and retry
                messages.append({
                    "role": "system",
                    "content": (
                        f"Your previous response was invalid JSON or did not follow the schema. "
                        f"Please output ONLY a valid JSON object of the form: "
                        f"{{'tool': <string>, 'params': <object>}} with double quotes."
                    )
                })
                continue
            else:
                # Final fallback
                return json.dumps({"tool": "none", "params": {}, "error": str(e)})


# ----------------------------------------------------------------------
# Utility: Normalize tool schema
# ----------------------------------------------------------------------
def normalize_tool_json(raw: str) -> dict:
    """Force output to use {tool, params} keys only."""
    cleaned = raw.strip().replace("'", '"')

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"tool": "none", "params": {}}

    #  Normalize synonyms
    if "action" in data and "tool" not in data:
        data["tool"] = data.pop("action")

    if "parameters" in data and "params" not in data:
        data["params"] = data.pop("parameters")

    #  Handle missing params key
    if "params" not in data:
        data["params"] = {}

    return data


# ----------------------------------------------------------------------
# Utility: Call MCP tool endpoint
# ----------------------------------------------------------------------
async def call_mcp_tool(tool_name, params):
    """Call the MCP tool endpoint via REST."""
    async with ClientSession() as session:
        try:
            url = f"{MCP_SERVER_URL}/{tool_name}"
            async with session.post(url, json=params, timeout=30) as resp:
                if resp.status != 200:
                    return f" MCP tool error: {resp.status} {await resp.text()}"
                return await resp.text()
        except Exception as e:
            return f" Exception while calling MCP tool: {e}"


# ----------------------------------------------------------------------
# Chatbot query handler
# ----------------------------------------------------------------------
async def handle_query(user_message):
    """
    Main orchestrator:
      1. Ask GPT which MCP tool to call (in JSON mode)
      2. Normalize and validate tool info
      3. Call MCP tool
      4. Return results
    """
    # Step 1: Ask GPT which tool to use
    tool_prompt = [
        {
            "role": "system",
            "content": (
                "You are a FinOps assistant. Choose the correct MCP tool to answer the user's question. "
                "Output ONLY valid JSON (no markdown, no commentary) in this exact format:\n"
                "{ \"tool\": <string>, \"params\": <object> }\n\n"
                f"Available tools:\n{json.dumps(MCP_TOOLS, indent=2)}"
            )
        },
        {"role": "user", "content": user_message}
    ]

    raw_tool_resp = await send_to_llm(tool_prompt, expect_json=True)
    tool_info = normalize_tool_json(raw_tool_resp)

    tool_name = tool_info.get("tool", "none")
    params = tool_info.get("params", {})

    if tool_name == "none":
        return " Could not determine which MCP tool to call."

    # Step 2: Call the selected MCP tool
    tool_result = await call_mcp_tool(tool_name, params)

    # Step 3: Combine results
    return f" Tool Selected: {tool_name}\n Params: {json.dumps(params)}\n\n Result:\n{tool_result}"


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def init_streamlit_ui():
    st.set_page_config(page_title="FinOps Chatbot", page_icon="🤖", layout="centered")
    st.title("💬 FinOps Chatbot (MCP-Aware)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about FinOps data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = asyncio.run(handle_query(user_input))
                st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})


# ----------------------------------------------------------------------
# Run app
# ----------------------------------------------------------------------
if __name__ == "__main__":
    init_streamlit_ui()
