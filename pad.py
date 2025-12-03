clean_json = llm_decision.strip()
if "```" in clean_json:
    clean_json = clean_json.replace("```json", "").replace("```", "")
parsed = json.loads(clean_json)


import json, ast, re

def safe_parse_llm_output(raw_text: str):
    """
    Safely parse LLM JSON-like output that may contain single quotes or markdown formatting.
    Supports both JSON and Python dict syntax.
    """
    if not raw_text:
        raise ValueError("Empty LLM output")

    # Clean code fences and whitespace
    clean = raw_text.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    # If multiple JSON/dict-like blocks, take the last one
    matches = re.findall(r"\{.*\}", clean, flags=re.DOTALL)
    if matches:
        clean = matches[-1]




async def ask_llm(user_input: str, history: list) -> str:
    """
    Ask LLM to decide which workflow or tool to execute with conversation + tool context awareness.
    """

    system_prompt = """
    You are the Orchestrator for the Astra MCP Server system...
    (your existing prompt unchanged)
    """

    # Build correct message list
    messages = [{"role": "system", "content": system_prompt}]

    # Include last 6 valid chat messages only
    for msg in history[-6:]:
        if "role" in msg and "content" in msg:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current user turn
    messages.append({"role": "user", "content": user_input})

    # Call LLM
    response = await llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.2,
    )

    output = response.choices[0].message.content.strip()

    # Normalize JSON response
    cleaned = output.replace("```json", "").replace("```", "").strip()

    return json.loads(cleaned)


    # Try strict JSON parsing
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Try Python literal evaluation (handles single quotes)
    try:
        return ast.literal_eval(clean)
    except Exception as e:
        raise ValueError(f"Unable to parse LLM output: {e}\nRaw text: {raw_text}")

parsed = safe_parse_llm_output(llm_decision)


async def ask_llm(user_input: str) -> str:
    """Ask LLM to decide which tool to call or return a message."""
    system_prompt = """
You are the Orchestrator for the Unified MCP Server that controls FinOps and GitHub tools.

Your job:
- Decide which MCP tool (if any) to invoke based on the user's request.
- If a tool applies, return its name and parameters.
- If not, return a human-friendly message instead.

Available tools:
1. get_rightsizing_recommendations(environment, resource_id)
2. validate_cost_savings(recommendation_id)
3. estimate_savings(resource_id)
4. clone_repo(repo_url)
5. create_pull_request(branch_name, title, body)
6. commit_and_push(commit_message)

Output format (JSON or Python dict):
- If a tool applies:
  {"tool": "tool_name", "params": {"param1": "value1"}}

- If no tool applies:
  {"tool": null, "message": "Your friendly message here"}

Examples:
User: "Get rightsizing for prod appsvc"
→ {"tool": "get_rightsizing_recommendations", "params": {"environment": "prod", "resource_id": "appsvc"}}

User: "hi"
→ {"tool": null, "message": "Hi there! I can help you run GitHub or FinOps operations such as fetching cost recommendations or creating pull requests."}

User: "what can you do"
→ {"tool": null, "message": "I can perform FinOps analytics and GitHub operations via the MCP tools."}

Always output exactly one JSON or Python dict block.
"""

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def safe_parse_llm_output(output: str):
    """Safely parse LLM output into a dict."""
    output = output.strip()
    try:
        # Handle JSON-like or Python dict-like output
        cleaned = output.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            # Try evaluating as Python dict (single quotes)
            return eval(cleaned)
        except Exception:
            return {"tool": None, "message": f"Could not parse LLM output: {output}"}


# --- Streamlit UI ---
st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input (async)
if prompt := st.chat_input("Ask me to perform FinOps or GitHub actions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm_decision = await ask_llm(prompt)
            parsed = safe_parse_llm_output(llm_decision)

            tool_name = parsed.get("tool")
            params = parsed.get("params", {})
            message = parsed.get("message")

            if tool_name and tool_name != "none":
                # Call the appropriate MCP tool
                result = await call_mcp_tool(tool_name, params)
                st.write(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(result)}
                )
            else:
                # Show only the friendly message, not the full dict
                friendly_msg = message or "I couldn’t find a suitable tool for your request."
                st.write(friendly_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": friendly_msg}
                )

=====================================

# --- Main orchestrator logic ---
async def orchestrate(user_input: str):
    """Core orchestration logic for the chatbot."""
    llm_decision = await ask_llm(user_input)
    parsed = safe_parse_llm_output(llm_decision)

    tool = parsed.get("tool")
    params = parsed.get("params", {})
    message = parsed.get("message")

    # Case 1: Friendly message only
    if tool == "none":
        return message or "Hi there! 👋 How can I help you today?"

    # Case 2: Tool found → execute via MCP
    if tool:
        result = await call_mcp_tool(tool, params)
        return result

    # Case 3: Fallback
    return "I couldn’t determine which tool to use."


# --- Streamlit UI ---
st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question or request..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = asyncio.run(orchestrate(prompt))

            # Display only the message if tool was "none"
            st.write(result)
            st.session_state.messages.append({"role": "assistant", "content": result})




if result.get("tool") == "none":
                # Friendly / fallback message
                st.write(result.get("message"))
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.get("message")}
                )

            elif isinstance(result.get("result"), dict):
                data = result["result"].get("data", result["result"])
                if "columns" in data and "rows" in data:
                    # Display clean table
                    st.write(f"**Results from `{result['tool']}`:**")
                    st.dataframe(data["rows"], columns=data["columns"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Displayed table for {result['tool']}"}
                    )
                else:
                    st.write(str(data))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": str(data)}
                    )
            else:
                st.write(str(result))
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(result)}
                )






# --- Main orchestrator logic ---
async def orchestrate(user_input: str):
    """Core orchestration logic for the chatbot."""
    llm_decision = await ask_llm(user_input)
    parsed = safe_parse_llm_output(llm_decision)

    tool = parsed.get("tool")
    params = parsed.get("params", {})
    message = parsed.get("message")

    # Case 1: Friendly message only
    if tool == "none":
        return {"tool": "none", "message": message or "Hi there!  How can I help you today?"}

    # Case 2: Tool found → execute via MCP
    if tool:
        try:
            result = await call_mcp_tool(tool, params)
            return {"tool": tool, "result": result}
        except Exception as e:
            return {"tool": tool, "message": f" Tool execution failed: {str(e)}"}

    # Case 3: Fallback — always return dict, not string
    return {"tool": "none", "message": "I couldn’t determine which tool to use."}




system_prompt = """
You are the Orchestrator for the Unified MCP Server system.

You can decide between:
1. Single-tool operations
2. The special 'update_rightsize_workflow'
3. General knowledge responses (for normal user questions)

If user asks something like:
"update the right-size variables in repository X for environment Y",
return this JSON:
{
  "workflow": "update_rightsize_workflow",
  "params": {
      "repo": "repo_123",
      "environment": "env"
  }
}

If user request clearly maps to a single tool (like 'clone repo', 'list branches', 'get recommendations'):
{
  "workflow": "single_tool",
  "tool": "tool_name",
  "params": {...}
}

If user asks a general or unrelated question (like "what is Terraform?", "how does Git work?", "who are you?"):
{
  "workflow": "general_question",
  "answer": "Your normal natural-language answer here."
}

If user just greets:
{
  "workflow": "none",
  "message": "Hi there! 👋 I can help you automate FinOps and Infra updates."
}

Always respond with strict JSON — no markdown, no commentary.
"""

elif workflow == "general_question":
    return llm_output.get("answer")




My mcp orchestrator code 'app.py' and 'mcp_server.py' code are given below. List down all the pytest-based automated tests I need to write for the MCP server and orchestrator for CI/CD readiness and for deploying to Azure DevOps Pipelines or GitHub Actions with Azure. Orchestrator 'app.py' code is like the following -

Generate a ready-to-run tests/ folder scaffold (with actual pytest code templates, mocks, and fixtures for all these categories). Give me the actual code for all 26 tests so I can plug it into Azure DevOps or GitHub Action pipeline


# ================================
# MCP SERVER DOCKERFILE
# ================================

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install git + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    unixodbc-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose MCP server port (default 8100)
EXPOSE 8100

# Start MCP server
# Assumes you’re using FastMCP or similar pattern
CMD ["python", "server.py"]


# ================================
# ORCHESTRATOR (STREAMLIT) DOCKERFILE
# ================================

FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Streamlit defaults
EXPOSE 8501

# Set environment (Azure OpenAI keys, MCP URL, etc.)
# These are injected by CI/CD pipeline or docker-compose
ENV STREAMLIT_SERVER_PORT=8501

# Optional: prevent Streamlit from asking for email at runtime
ENV STREAMLIT_TELEMETRY=False
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=False

# Command to run orchestrator app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


version: "3.9"

services:
  mcp-server:
    build:
      context: ./mcp-server
      dockerfile: Dockerfile.mcp
    container_name: finops-mcp
    ports:
      - "8100:8100"
    environment:
      - PYTHONUNBUFFERED=1

  orchestrator:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile.orchestrator
    container_name: finops-orchestrator
    ports:
      - "8501:8501"
    depends_on:
      - mcp-server
    environment:
      - FINOPS_GITHUB_MCP_URL=http://mcp-server:8100/mcp
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_DEPLOYMENT=gpt-4.1


# Install ODBC and OS deps
RUN apt-get update && apt-get install -y curl apt-transport-https gnupg unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/12/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


streamlit==1.36.0
pyarrow==14.0.2


# Log this activity
mcp_logs.append({"tool": tool, "params": params, "status": "Success"})




async def orchestrate(user_input: str, mcp_logs: list, messages: list):
    """Core orchestration logic with context-aware chaining."""
    # Step 1: Ask LLM for decision
    llm_decision = await ask_llm(user_input, messages)
    parsed = safe_parse_llm_output(llm_decision)

    tool = parsed.get("tool")
    params = parsed.get("params", {})
    message = parsed.get("message")

    # Step 2: Friendly message only
    if tool == "none":
        return {"tool": "none", "message": message or "Hi there! How can I help you today?"}

    # Step 3: Tool found → execute via MCP
    if tool:
        result = await call_mcp_tool(tool, params)

        # Record tool execution in logs
        mcp_logs.append({
            "tool": tool,
            "params": params,
            "status": "Success",
            "result": result
        })

        # Feed this tool result back into conversation
        messages.append({
            "role": "assistant",
            "content": f"Tool `{tool}` executed with params {params}. Result:\n{json.dumps(result, indent=2)}"
        })

        return result

    # Step 4: Fallback
    return {"tool": "none", "message": "I couldn’t determine which tool to use."}



async def ask_llm(user_input: str, messages: list) -> str:
    system_prompt = """
You are the Orchestrator for the Unified MCP Server that controls FinOps and GitHub tools.

You can reason over conversation history, including tool results.

If a previous tool result is present, use it to generate insights or decide the next tool.

Examples:
- If user says "summarize that result", read the previous tool output.
- If user says "run cost validation on that resource", extract resource_id from previous tool output.

Always return **one JSON** object:
{"tool": "tool_name", "params": {...}}
or
{"tool": "none", "params": {}, "message": "text response"}
"""

    # Include last few turns for context (with tool results)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in messages[-6:]]
    chat_history.append({"role": "user", "content": user_input})

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": system_prompt}] + chat_history,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()




from pydantic import BaseModel

def serialize_result(result):
    """Safely convert MCP or Pydantic results to JSON-serializable objects."""
    # If result is a Pydantic model
    if isinstance(result, BaseModel):
        return result.model_dump()

    # If result looks like an MCP-style object with 'output' or 'error' attributes
    if hasattr(result, "output") or hasattr(result, "error"):
        return {
            "output": serialize_result(getattr(result, "output", None)),
            "error": getattr(result, "error", None)
        }

    # If it's a dict, list, or primitive
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result

    # Fallback: convert to string
    return str(result)



serialized_result = serialize_result(result)

messages.append({
    "role": "assistant",
    "content": (
        f"Tool `{tool}` executed with params {params}. "
        f"Result:\n{json.dumps(serialized_result, indent=2)}"
    )
})




async def ask_llm(user_input: str, history: list) -> str:
    """Ask LLM with conversation + tool context awareness."""
    system_prompt = """
You are the Orchestrator for the Unified MCP Server that controls FinOps and GitHub tools.

You maintain conversational memory — use the chat history to understand user intent.

If the user refers to something mentioned earlier (like “summarize that”, “use the last output”, or “continue from there”), 
you must interpret it in light of the previous tool result or system message.

If user greetings or general chat:
  Return: {"tool": "none", "params": {}, "message": "Hi there! 👋 I can help you with GitHub and FinOps operations."}

Otherwise, choose one of the available tools:
1. get_rightsizing_recommendations(environment, resource_id)
2. validate_cost_savings(recommendation_id)
3. estimate_savings(resource_id)
4. clone_repo(repo_url)
5. create_pull_request(branch_name, title, body)
6. commit_and_push(commit_message)

Respond strictly in JSON format only:
{"tool": "tool_name", "params": {"param1": "value1"}}
"""

    # Build message list with full conversation context
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:  # include last ~6 turns for compact context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()



if history and "Tool `" in history[-1]["content"]:
    messages.append({
        "role": "system",
        "content": f"Previous tool output summary:\n{history[-1]['content'][:1000]}"
    })



SELECT 
    T1.*, 
    T2.*
FROM 
    T1
JOIN 
    T2
    ON JSON_VALUE(T1.tags, '$.id') = T2.id;


import json, re, astimport json, re, ast

def safe_parse_llm_output(raw_text: str):
    """Parse LLM output safely into a consistent schema for the orchestrator."""
    # Handle None or non-string input
    if not raw_text or not isinstance(raw_text, str):
        return {"tool": "none", "params": {}, "message": "Empty response from LLM."}

    clean = raw_text.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    # Try extracting a JSON-like block
    matches = re.findall(r"\{.*?\}", clean, flags=re.DOTALL)
    if matches:
        clean = matches[-1]  # take the last block

    # 1️. Try strict JSON parsing
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list) and parsed and isinstance(parsed[-1], dict):
            return parsed[-1]
    except json.JSONDecodeError:
        pass

    # 2️. Try Python literal eval (handles single quotes)
    try:
        parsed = ast.literal_eval(clean)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list) and parsed and isinstance(parsed[-1], dict):
            return parsed[-1]
    except Exception:
        pass

    # 3️. If it's plain text (no JSON at all)
    if not ("{" in clean and "}" in clean):
        return {"tool": "none", "params": {}, "message": clean}

    # 4️. Final fallback — never break caller expectations
    return {"tool": "none", "params": {}, "message": f"Unrecognized LLM output: {raw_text}"}



async def run_proactive_stage(latest_user_msg: str, assistant_response: str, history: list):
    """
    Takes the assistant's response and asks the model:
    - Should I suggest a next step?
    - Should I remind about required parameters?
    - Should I propose continuing the workflow?
    """

    proactive_prompt = f"""
    You are the secondary reasoning module of a 2-stage proactive assistant.
    The user said: {latest_user_msg}
    The assistant responded: {assistant_response}

    Based on BOTH, decide if you should propose a proactive follow-up.
    Proactive suggestions include:
      - relevant next-step actions
      - recommended workflows
      - useful follow-up insights the user didn’t ask but would benefit from
    
    Respond STRICTLY in JSON:
    {{
       "should_suggest": true/false,
       "suggestion": "text of the suggestion (empty if none)"
    }}
    """

    response = await llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": proactive_prompt},
        ],
        temperature=0.2,
    )

    parsed = json.loads(response.choices[0].message.content)
    return parsed


# Run the proactive LLM stage
proactive = await run_proactive_stage(user_input, final_answer, st.session_state.chat_history)

if proactive.get("should_suggest"):
    final_answer += "\n\n *Suggested next step:* " + proactive["suggestion"]



I have some code below for an MCP application including a finops+github+infra MCP server, a client (orchestrator agent) and a chatbot (in streamlit).
It can support single MCP tool calls as well as workflows. 
However, my workflow "run_update_right_size_workflow" is written in plain python. 
I want to revamp the code to do multi-step proactive workflow orchestration using LangChain + LangGraph. 
Guide me best approach to do it.
Below, I am pasting my "MCP Server", "Orchestrator" and "Streamlit UI" codes for reference.



# orchestrator/workflow_rightsize.py
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import StructuredTool
from orchestrator.utils import call_mcp_tool
import asyncio
import os

# --- MCP URLs ---
MCP_INFRA_URL = os.getenv("INFRA_MCP_URL", "http://localhost:8200/mcp")
MCP_OPS_FINOPS_URL = os.getenv("OPS_FINOPS_MCP_URL", "http://localhost:8100/mcp")


# ---------- Wrap each MCP call as LangChain Tools ----------
clone_tool = StructuredTool.from_function(
    func=lambda repo_url, branch="main": asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "clone_repo", {"repo_url": repo_url, "branch": branch})
    ),
    name="clone_repo",
    description="Clone GitHub repository"
)

finops_tool = StructuredTool.from_function(
    func=lambda environment, humanaID: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "get_rightsizing_recommendations",
                      {"environment": environment, "humanaID": humanaID})
    ),
    name="get_rightsizing_recommendations",
    description="Fetch right-sizing recommendations from FinOps DB"
)

analyze_tool = StructuredTool.from_function(
    func=lambda repo_path: asyncio.run(
        call_mcp_tool(MCP_INFRA_URL, "analyze_tf_file", {"repo_path": repo_path})
    ),
    name="analyze_tf_file",
    description="Analyze Terraform files and identify right-size variables"
)

update_tool = StructuredTool.from_function(
    func=lambda repo_path, analysis_json, recommendations: asyncio.run(
        call_mcp_tool(MCP_INFRA_URL, "update_tf_file", {
            "repo_path": repo_path,
            "analysis_json": analysis_json,
            "recommendations": recommendations
        })
    ),
    name="update_tf_file",
    description="Update Terraform files using FinOps recommendations"
)

commit_tool = StructuredTool.from_function(
    func=lambda repo_path, branch, message: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "commit_and_push", {
            "repo_path": repo_path,
            "branch": branch,
            "commit_message": message
        })
    ),
    name="commit_and_push",
    description="Commit and push updates"
)

pr_tool = StructuredTool.from_function(
    func=lambda repo_url, source_branch, title, body: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "create_pull_request", {
            "repo_url": repo_url,
            "source_branch": source_branch,
            "title": title,
            "body": body
        })
    ),
    name="create_pull_request",
    description="Create pull request"
)

cleanup_tool = StructuredTool.from_function(
    func=lambda local_path: asyncio.run(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "cleanup_clone", {"local_path": local_path})
    ),
    name="cleanup_clone",
    description="Cleanup temporary cloned repo"
)


# ---------- Build the LangGraph ----------
def build_rightsize_workflow():
    graph = StateGraph()

    graph.add_node("clone_repo", ToolNode(tool=clone_tool))
    graph.add_node("finops", ToolNode(tool=finops_tool))
    graph.add_node("analyze", ToolNode(tool=analyze_tool))
    graph.add_node("update", ToolNode(tool=update_tool))
    graph.add_node("commit", ToolNode(tool=commit_tool))
    graph.add_node("create_pr", ToolNode(tool=pr_tool))
    graph.add_node("cleanup", ToolNode(tool=cleanup_tool))

    # --- Parallel: clone_repo + finops ---
    graph.add_edge("clone_repo", "analyze")
    graph.add_edge("finops", "analyze")

    # --- Sequential chain ---
    graph.add_edge("analyze", "update")
    graph.add_edge("update", "commit")
    graph.add_edge("commit", "create_pr")
    graph.add_edge("create_pr", "cleanup")

    # --- Error handling: cleanup on failure ---
    def on_error(e, state):
        state["error"] = str(e)
        return "cleanup"

    graph.on_error(on_error)
    return graph.compile()



from langchain.tools import StructuredTool
import asyncio

def mcp_tool_wrapper(
    tool_name: str,
    param_schema: dict,
    description: str,
    mcp_url: str,
):
    """
    Wrap an MCP tool as a LangChain StructuredTool.

    Args:
        tool_name (str): Name of the MCP tool to call on the MCP server.
        param_schema (dict): Dict of parameter names with default values (used to build signature).
        description (str): Description for LangChain.
        mcp_url (str): MCP server endpoint.

    Returns:
        StructuredTool: LangChain tool ready to use.
    """

    async def _run_mcp(**kwargs):
        from orchestrator_core import call_mcp_tool   # or your actual import
        return await call_mcp_tool(tool_name, kwargs)

    # Sync wrapper required because LangChain tools must be sync
    def sync_wrapper(**kwargs):
        return asyncio.run(_run_mcp(**kwargs))

    # Build a function signature dynamically from param_schema
    sync_wrapper.__annotations__ = {k: type(v) for k, v in param_schema.items()}
    sync_wrapper.__name__ = tool_name

    return StructuredTool.from_function(
        func=sync_wrapper,
        name=tool_name,
        description=description,
    )


clone_tool = mcp_tool_wrapper(
    tool_name="clone_repo",
    param_schema={"repo_url": "", "branch": "main"},
    description="Clone a GitHub repository.",
    mcp_url=MCP_URL,
)



# -----------------------------------------
# 3️. ROUTER NODE — END EARLY LOGIC (FIXED)
# -----------------------------------------
def check_recommendations(state: RightSizeState):
    """Router must always return a dict, not a string."""
    recos = state.recommendations

    if recos == [{}] or recos == []:
        return {"route": "END"}

    return {"route": "update_infra"}


# -----------------------------------------
# 4️. GRAPH DEFINITION (FIXED CONDITIONAL)
# -----------------------------------------
def build_rightsize_graph():
    graph = StateGraph(RightSizeState)

    # register nodes...

    graph.add_conditional_edges(
        "check_recommendations",
        lambda st: st["route"],
        {
            "END": END,
            "update_infra": "update_infra",
        }
    )

    # Continue with edges...


graph.add_conditional_edges(
    "check_recommendations",
    lambda st: st.route,
    {
        "update_infra": "update_infra",
        "no_changes": END
    }
)






=================


async def ask_llm(user_input: str, history: list) -> str:
    """
    Ask LLM to decide which workflow or tool to execute with conversation + tool context awareness.
    """

    system_prompt = """
    You are the Orchestrator for the Astra MCP Server system...
    (your existing prompt unchanged)
    """

    # Build correct message list
    messages = [{"role": "system", "content": system_prompt}]

    # Include last 6 valid chat messages only
    for msg in history[-6:]:
        if "role" in msg and "content" in msg:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current user turn
    messages.append({"role": "user", "content": user_input})

    # Call LLM
    response = await llm_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.2,
    )

    output = response.choices[0].message.content.strip()

    # Normalize JSON response
    cleaned = output.replace("```json", "").replace("```", "").strip()

    return json.loads(cleaned)

================================

# ============================================================
# Custom MCP HTTP Client (for listing tools, manual calls)
# ============================================================
import aiohttp

class McpHttpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def list_tools(self):
        """
        Fetch all tools exposed by the MCP server.
        Expects MCP Server exposes GET /tools
        """
        async with self.session.get(f"{self.base_url}/tools") as resp:
            return await resp.json()

    async def call_tool(self, tool_name: str, params: dict):
        """
        Call a specific MCP tool.
        Expects POST /tools/<tool_name>
        """
        async with self.session.post(
            f"{self.base_url}/tools/{tool_name}", json=params
        ) as resp:
            return await resp.json()

    async def close(self):
        await self.session.close()

==============================

@app.get("/show_tools")
async def show_tools():
    client = McpHttpClient(MCP_URL.replace("/mcp", ""))   # Point directly to server root
    tools = await client.list_tools()
    await client.close()
    return {"tools": tools}

===============================

from fastapi import FastAPI

app = FastAPI(title="Astra MCP Server")

# Example tool registry (fastmcp handles actual MCP tools, we keep REST metadata)
REGISTERED_TOOLS = {}


# When defining tools using fastmcp decorator:
from fastmcp import tool

@tool
async def clone_repo(repo_url: str, branch: str):
    ...
# Register for REST listing
REGISTERED_TOOLS["clone_repo"] = {
    "description": "Clone a git repo",
    "params": ["repo_url", "branch"]
}


@tool
async def get_rightsizing_recommendations(environment: str, app_id: str):
    ...
REGISTERED_TOOLS["get_rightsizing_recommendations"] = {
    "description": "Get finops rightsize recommendations",
    "params": ["environment", "app_id"]
}


# ============================================================
# REST ENDPOINT: GET /tools
# ============================================================
@app.get("/tools")
async def list_tools():
    """
    REST endpoint for orchestrator. Returns all tools.
    """
    return {"tools": REGISTERED_TOOLS}


# OPTIONAL: Call tool via REST (useful for testing)
@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, params: dict):
    """
    Allows REST clients (your orchestrator) to call MCP tools.

    But internally we still use fastmcp tool execution.
    """
    if tool_name not in REGISTERED_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    # Import the tool function dynamically
    fn = globals().get(tool_name)
    if not fn:
        return {"error": f"Tool not found in globals: {tool_name}"}

    result = await fn(**params)
    return {"result": result}


from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool
from langchain.chains import LLMChain
from langchain.agents import load_tools

# ---------------------------------------------------------
# 1. Wrap your two update functions as LangChain tools
# ---------------------------------------------------------

@tool("locals_updater")
def locals_updater_tool(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform LOCALS file using update_terraform_file_locals_type()."""
    return update_terraform_file_locals_type(terraform_file, metadata_file, updates, similarity_threshold)


@tool("yaml_updater")
def yaml_updater_tool(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform YAML file using update_terraform_file_yaml_type()."""
    return update_terraform_file_yaml_type(terraform_file, metadata_file, updates, similarity_threshold)


# ---------------------------------------------------------
# 2. Create the sub-agents (each one bound to its tool)
# ---------------------------------------------------------

llm = OpenAI(temperature=0)

locals_updater_agent = initialize_agent(
    tools=[locals_updater_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

yaml_updater_agent = initialize_agent(
    tools=[yaml_updater_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ---------------------------------------------------------
# 3. Router agent prompt (LLM decides which sub-agent to use)
# ---------------------------------------------------------

ROUTER_PROMPT = """
You are the updater_agent.

Your job: choose which updater sub-agent to call based on user instructions.

Use the rules:

- If the Terraform file uses `locals { ... }` or is an HCL file → CALL `locals_updater`.
- If the Terraform file is YAML → CALL `yaml_updater`.

Return ONLY the tool call. Do NOT answer directly.

User request:
{input}
"""

router_prompt = PromptTemplate(
    input_variables=["input"],
    template=ROUTER_PROMPT
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

# ---------------------------------------------------------
# 4. Router Agent Logic (custom)
# ---------------------------------------------------------

def router_agent(user_input: str):
    """LLM chooses the correct sub-agent, then executes it."""
    decision = router_chain.run(user_input)

    if "locals_updater" in decision:
        print("\n🔧 Router selected: locals_updater_agent\n")
        return locals_updater_agent.run(user_input)

    if "yaml_updater" in decision:
        print("\n🔧 Router selected: yaml_updater_agent\n")
        return yaml_updater_agent.run(user_input)

    return "Router could not determine which updater to use."


# ---------------------------------------------------------
# 5. Example Usage
# ---------------------------------------------------------

user_query = """
Please update this locals.tf file using metadata.json.
Updates list:
[
  {"environment_name": "dev", "resource_group_name": "rg1", "resource_name": "sql", "variable_name": "max-size-gb", "variable_value": 50}
]
terraform_file="locals.tf"
metadata_file="metadata.json"
"""

response = router_agent(user_query)
print(response)




from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Your existing update functions
from your_file import (
    update_terraform_file_locals_type,
    update_terraform_file_yaml_type
)

# ---------------------------------------------------------
# 1. Wrap your two update functions as LangChain tools
# ---------------------------------------------------------

@tool
def locals_updater(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform LOCALS file."""
    return update_terraform_file_locals_type(terraform_file, metadata_file, updates, similarity_threshold)


@tool
def yaml_updater(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform YAML file."""
    return update_terraform_file_yaml_type(terraform_file, metadata_file, updates, similarity_threshold)


TOOLS = {
    "locals_updater": locals_updater,
    "yaml_updater": yaml_updater,
}


# ---------------------------------------------------------
# 2. LLM used everywhere
# ---------------------------------------------------------

llm = ChatOpenAI(model="gpt-4.1", temperature=0)


# ---------------------------------------------------------
# 3. Router prompt
# ---------------------------------------------------------

ROUTER_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
You are a routing agent.

Decide which updater tool the user needs.

Rules:
- If the file is HCL or contains 'locals {{ ... }}' → choose "locals_updater"
- If the file is YAML → choose "yaml_updater"

Return ONLY one word:
locals_updater
OR
yaml_updater

User request:
{input}
"""
)


router_chain = (
    ROUTER_PROMPT
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------
# 4. Router agent wrapper
# ---------------------------------------------------------

def router_agent(user_input: str):
    """Decide which updater tool to invoke."""
    decision = router_chain.invoke({"input": user_input}).strip()

    print(f"\n Router decision → {decision}\n")

    if decision not in TOOLS:
        return f"Router error: unknown tool '{decision}'"

    # Run actual tool
    tool_fn = TOOLS[decision]

    # Extract arguments from user input (LLM can help)
    # Quick heuristic: we run a small parser prompt
    parse_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
Extract the following fields from the text and return ONLY JSON:

- terraform_file
- metadata_file
- updates (array)

Text:
{input}
"""
    )

    parsed = (
        parse_prompt
        | llm
        | StrOutputParser()
    ).invoke({"input": user_input})

    import json
    args = json.loads(parsed)

    # Call tool with proper arguments
    return tool_fn.invoke(args)


# ---------------------------------------------------------
# 5. Example
# ---------------------------------------------------------

query = """
Please update this locals.tf file using metadata.json.
Updates list:
[
  {"environment_name": "dev", "resource_group_name": "rg1",
   "resource_name": "sql", "variable_name": "max-size-gb", "variable_value": 50}
]
terraform_file="locals.tf"
metadata_file="metadata.json"
"""

print(router_agent(query))



yaml_updater_tool_lc = StructuredTool.from_function(
    name="yaml_updater_tool",
    description="Updates specified fields inside a YAML file.",
    func=lambda yaml_file, metadata_file, updates: yaml_updater_tool(
        yaml_file, metadata_file, updates
    ),
    args_schema=YamlUpdaterInput,
)


# mcp_tool_loader.py

import json
import asyncio
from fastmcp import HTTPClient
from langchain.tools import StructuredTool
from pydantic import create_model, BaseModel

MCP_SERVER_URL = "http://mcp-server:8000"


async def load_mcp_tools():
    """Connect to MCP server and load tools as LangChain StructuredTool objects."""
    client = HTTPClient(MCP_SERVER_URL)
    await client.connect()

    discovery = await client.list_tools()
    tools = []

    for tool_def in discovery.tools:
        tool_name = tool_def.name
        tool_description = tool_def.description or f"MCP tool {tool_name}"

        # ---- 1. Build args_schema from MCP tool parameters ----
        # Each MCP tool declares its parameters as JSON schema
        schema_props = tool_def.inputSchema.get("properties", {})
        required = tool_def.inputSchema.get("required", [])

        # Dynamically create a Pydantic model
        fields = {}
        for param_name, param_info in schema_props.items():
            # Try best-effort type detection
            json_type = param_info.get("type", "string")
            if json_type == "string":
                py_type = (str, ...)
            elif json_type == "number":
                py_type = (float, ...)
            elif json_type == "integer":
                py_type = (int, ...)
            elif json_type == "boolean":
                py_type = (bool, ...)
            elif json_type == "array":
                py_type = (list, ...)
            elif json_type == "object":
                py_type = (dict, ...)
            else:
                py_type = (str, ...)

            # Required field marker
            if param_name not in required:
                py_type = (py_type[0], None)

            fields[param_name] = py_type

        ArgsSchema = create_model(
            f"{tool_name}_schema",
            **fields,
            __base__=BaseModel
        )

        # ---- 2. MCP async caller ----
        async def _mcp_caller(**kwargs):
            result = await client.call_tool(tool_name, kwargs)
            return result.content

        # Wrap async function for StructuredTool (must sync)
        def sync_wrapper(**kwargs):
            return asyncio.run(_mcp_caller(**kwargs))

        # ---- 3. Build StructuredTool ----
        tool = StructuredTool.from_function(
            name=tool_name,
            description=tool_description,
            func=sync_wrapper,
            args_schema=ArgsSchema,
        )

        tools.append(tool)

    return tools




# mcp_tool_loader.py

import json
import asyncio
from fastmcp import HTTPClient
from pydantic import BaseModel, create_model
from langchain.tools import StructuredTool

MCP_SERVER_URL = "http://mcp-server:8000"


async def load_mcp_tools():
    client = HTTPClient(MCP_SERVER_URL)
    await client.connect()

    discovery = await client.list_tools()
    tools = []

    for tool_def in discovery.tools:
        tool_name = tool_def.name
        tool_description = tool_def.description or f"MCP tool {tool_name}"

        # --- Build args schema (dynamic Pydantic model) ---
        schema_props = tool_def.inputSchema.get("properties", {})
        required = tool_def.inputSchema.get("required", [])

        fields = {}
        for param_name, param_info in schema_props.items():
            json_type = param_info.get("type", "string")

            if json_type == "string":
                annotation = str
            elif json_type == "integer":
                annotation = int
            elif json_type == "number":
                annotation = float
            elif json_type == "boolean":
                annotation = bool
            elif json_type == "array":
                annotation = list
            elif json_type == "object":
                annotation = dict
            else:
                annotation = str

            if param_name in required:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, None)

        ArgsSchema = create_model(f"{tool_name}_schema", **fields)

        # --- Async MCP caller ---
        async def async_caller(**kwargs):
            result = await client.call_tool(tool_name, kwargs)
            return result.content

        # --- Sync wrapper SAFE inside running event loop ---
        def sync_wrapper(**kwargs):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop running → normal synchronous call
                return asyncio.run(async_caller(**kwargs))

            # Loop already running → schedule task & wait
            future = asyncio.run_coroutine_threadsafe(
                async_caller(**kwargs), loop
            )
            return future.result()

        # --- Create StructuredTool ---
        tool = StructuredTool.from_function(
            name=tool_name,
            description=tool_description,
            func=sync_wrapper,
            args_schema=ArgsSchema,
        )

        tools.append(tool)

    return tools




# mcp_server.py
import asyncio
from fastmcp import MCP, Tool, Schema

mcp = MCP("terraform-mcp")


# ----------------------------
# TOOL DEFINITIONS
# ----------------------------

@mcp.tool(
    name="locals_updater_tool",
    description="Updates a Terraform locals file with metadata values.",
    input_schema=Schema(
        {
            "terraform_file": {"type": "string"},
            "metadata_file": {"type": "string"},
            "updates": {"type": "array"},
            "similarity_threshold": {"type": "number"}
        },
        required=["terraform_file", "metadata_file", "updates"]
    )
)
async def locals_updater_tool(terraform_file, metadata_file, updates, similarity_threshold=0.75):
    # Place your business logic here
    return {
        "status": "success",
        "message": f"Updated locals in {terraform_file}",
        "updates": updates
    }


@mcp.tool(
    name="yaml_updater_tool",
    description="Updates a YAML file with metadata values.",
    input_schema=Schema(
        {
            "yaml_file": {"type": "string"},
            "metadata_file": {"type": "string"},
            "updates": {"type": "array"},
        },
        required=["yaml_file", "metadata_file", "updates"]
    )
)
async def yaml_updater_tool(yaml_file, metadata_file, updates):
    # Place your business logic here
    return {
        "status": "success",
        "message": f"Updated YAML file {yaml_file}",
        "updates": updates
    }


# ----------------------------
# START SERVER
# ----------------------------
if __name__ == "__main__":
    asyncio.run(mcp.run(host="0.0.0.0", port=8000))



# langchain_agent.py
import asyncio
import json
from pydantic import create_model
from fastmcp import HTTPClient

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType


MCP_SERVER_URL = "http://mcp-server:8000"


async def load_mcp_tools():
    """Discover MCP tools and wrap them as LangChain StructuredTool."""
    client = HTTPClient(MCP_SERVER_URL)
    await client.connect()

    discovery = await client.list_tools()
    tools = []

    for tool_def in discovery.tools:
        name = tool_def.name
        description = tool_def.description or f"MCP tool: {name}"

        # build pydantic args schema dynamically
        schema_props = tool_def.inputSchema.get("properties", {})
        required = tool_def.inputSchema.get("required", [])

        fields = {}
        for param_name, prop in schema_props.items():
            json_type = prop.get("type", "string")

            typemap = {
                "string": str,
                "number": float,
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            annotation = typemap.get(json_type, str)

            if param_name in required:
                fields[param_name] = (annotation, ...)
            else:
                fields[param_name] = (annotation, None)

        ArgsSchema = create_model(f"{name}_schema", **fields)

        # async→sync wrapper
        async def async_mcp_call(**kwargs):
            result = await client.call_tool(name, kwargs)
            return result.content

        def sync_wrapper(**kwargs):
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    async_mcp_call(**kwargs), loop
                )
                return future.result()
            except RuntimeError:
                return asyncio.run(async_mcp_call(**kwargs))

        tool = StructuredTool.from_function(
            name=name,
            description=description,
            func=sync_wrapper,
            args_schema=ArgsSchema,
        )

        tools.append(tool)

    return tools


# ----------------------------
# Run the agent
# ----------------------------

async def main():
    tools = await load_mcp_tools()

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    query = """
Update the YAML file c:\\xyz\\proj\\file.yaml 
using metadata c:\\xyz\\proj\\meta.json.
Updates list:
[
  {"resource": "abc", "value": 123}
]
"""

    result = agent.run(query)
    print("\nFINAL RESULT:\n", result)


if __name__ == "__main__":
    asyncio.run(main())





# mcp_server.py
import asyncio
from fastmcp import MCP

mcp = MCP("terraform-mcp")

# ----------------------------
# LOCALS TOOL
# ----------------------------

@mcp.tool(
    name="locals_updater_tool",
    description="Updates a Terraform locals file with metadata values.",
    input_schema={
        "type": "object",
        "properties": {
            "terraform_file": {"type": "string"},
            "metadata_file": {"type": "string"},
            "updates": {"type": "array"},
            "similarity_threshold": {"type": "number"},
        },
        "required": ["terraform_file", "metadata_file", "updates"]
    }
)
async def locals_updater_tool(terraform_file, metadata_file, updates, similarity_threshold=0.75):
    # Replace with your actual logic
    return {
        "status": "success",
        "message": f"Updated locals in {terraform_file}",
        "updates_count": len(updates),
    }


# ----------------------------
# YAML TOOL
# ----------------------------

@mcp.tool(
    name="yaml_updater_tool",
    description="Updates a YAML file with metadata values.",
    input_schema={
        "type": "object",
        "properties": {
            "yaml_file": {"type": "string"},
            "metadata_file": {"type": "string"},
            "updates": {"type": "array"},
        },
        "required": ["yaml_file", "metadata_file", "updates"]
    }
)
async def yaml_updater_tool(yaml_file, metadata_file, updates):
    # Replace with your actual logic
    return {
        "status": "success",
        "message": f"Updated YAML file {yaml_file}",
        "updates_count": len(updates),
    }


# ----------------------------
# START SERVER
# ----------------------------
if __name__ == "__main__":
    asyncio.run(mcp.run(host="0.0.0.0", port=8000))



888888888888888

# mcp_server.py
import asyncio
from fastmcp import MCP

mcp = MCP("terraform-mcp")

@mcp.tool()
async def locals_updater_tool(input: dict):
    """
    input = {
        "terraform_file": "...",
        "metadata_file": "...",
        "updates": [...],
        "similarity_threshold": 0.75
    }
    """

    terraform_file = input.get("terraform_file")
    metadata_file = input.get("metadata_file")
    updates = input.get("updates", [])
    threshold = input.get("similarity_threshold", 0.75)

    # call your real logic
    return {
        "status": "ok",
        "file": terraform_file,
        "updates_applied": len(updates),
        "threshold": threshold
    }


@mcp.tool()
async def yaml_updater_tool(input: dict):
    """
    input = {
        "yaml_file": "...",
        "metadata_file": "...",
        "updates": [...]
    }
    """

    yaml_file = input.get("yaml_file")
    metadata_file = input.get("metadata_file")
    updates = input.get("updates", [])

    return {
        "status": "ok",
        "file": yaml_file,
        "updates_applied": len(updates)
    }


if __name__ == "__main__":
    asyncio.run(mcp.run(host="0.0.0.0", port=8000))


# mcp_tool_loader.py
import json
from fastmcp import HTTPClient
from langchain_core.tools import StructuredTool

MCP_SERVER_URL = "http://mcp-server:8000"

async def load_mcp_tools():
    client = HTTPClient(MCP_SERVER_URL)
    await client.connect()

    discovery = await client.list_tools()
    tools = []

    for tool_def in discovery.tools:

        async def _caller(arg_dict, tool_name=tool_def.name):
            result = await client.call_tool(tool_name, arg_dict)
            return result.content

        # Every tool now takes ONE dict → so we wrap it with StructuredTool
        tools.append(
            StructuredTool.from_function(
                name=tool_def.name,
                description=tool_def.description,
                func=_caller,
            )
        )

    return tools

async def _caller(**kwargs):
    return await client.call_tool(name, kwargs)



mcp_client = None

async def load_mcp_tools():
    global mcp_client

    if mcp_client is None:
        mcp_client = Client(MCP_SERVER_URL)
        await mcp_client.connect()

    discovery = await mcp_client.list_tools()
    tools = []

    for tool_def in discovery:
        async def _caller(tool_name=tool_def.name, **kwargs):
            result = await mcp_client.call_tool(tool_name, kwargs)
            return result.content

        tools.append(
            StructuredTool.from_function(
                name=tool_def.name,
                description=tool_def.description or "",
                func=_caller,
            )
        )

    return tools



async def load_mcp_tools():
    async with Client(MCP_SERVER_URL) as client:
        discovery = await client.list_tools()
        tools = []

        for tool_def in discovery:

            async def _caller(input: dict, tool_name=tool_def.name):
                # caller accepts ONLY 'input' + captures tool_name via default
                async with Client(MCP_SERVER_URL) as c2:
                    result = await c2.call_tool(tool_name, {"arg_dict": input})
                    return result.content

            # IMPORTANT: define signature to SINGLE ARG
            _caller.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "input",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
            )

            tool = StructuredTool.from_function(
                name=tool_def.name,
                description=tool_def.description,
                func=_caller,
            )

            tools.append(tool)

    return tools

