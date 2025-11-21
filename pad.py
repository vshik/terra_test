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
