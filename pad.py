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
