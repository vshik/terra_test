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




import os
import subprocess
from urllib.parse import urlparse
from pydantic import BaseModel
from typing import Optional
from fastmcp import mcp, Context


class CloneInput(BaseModel):
    repo_url: str
    branch: Optional[str] = "main"
    base_dir: Optional[str] = "C:\\repos"  # Default clone base directory


class CloneOutput(BaseModel):
    local_path: str


@mcp.tool(name="clone_repo", description="Clone a GitHub repo to a specified local directory, keeping repo name as folder name")
def clone_repo(params: CloneInput, ctx: Context) -> CloneOutput:
    os.makedirs(params.base_dir, exist_ok=True)        # Ensure base directory exists    
    repo_name = os.path.splitext(os.path.basename(urlparse(params.repo_url).path))[0]    # Derive folder name from repo URL
    clone_path = os.path.join(params.base_dir, repo_name)
    
    if os.path.exists(clone_path):  # If folder already exists, handle gracefully
        # Optional: remove existing folder, or pull latest instead
        raise FileExistsError(f"The folder '{clone_path}' already exists. Remove it or specify another base_dir.")
    # Clone repo into named folder
    subprocess.run(["git", "clone", "-b", params.branch, params.repo_url, clone_path], check=True,)

    return CloneOutput(local_path=clone_path)



# --- Sidebar ---
with st.sidebar:
    st.header("📜 MCP Activity Log")
    if st.session_state.mcp_logs:
        for log in reversed(st.session_state.mcp_logs):
            st.markdown(f"**Tool:** `{log['tool']}`  \n**Params:** `{log['params']}`  \n**Status:** {log['status']}")
            st.divider()
    else:
        st.info("No MCP activity yet...")




# --- Add normal FastAPI endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"service": "FinOps GitHub MCP", "status": "running"}




# --- Define Input/Output Schemas ---
class AnalyzeInput(BaseModel):
    repo_url: str  # e.g., "https://github.com/org/repo_123.git"
    base_dir: Optional[str] = "C:\\repos"  # Default location for cloned repos
    branch: Optional[str] = "main"

class AnalyzeOutput(BaseModel):
    analysis_path: str
    local_repo_path: str


class UpdateInput(BaseModel):
    repo_path: str
    analysis_path: str
    recommendations: dict


class UpdateOutput(BaseModel):
    updated_files: list
    summary: str


# --- Initialize MCP server ---
mcp = MCP(name="InfraLogic MCP", description="Terraform Analysis and Update Tools")


# --- Utility: Clone repo if not already present ---
async def clone_repo_if_needed(repo_url: str, base_dir: str, branch: str = "main") -> str:
    """
    Clone a GitHub repo if not already cloned.
    Returns the local path to the repo.
    """
    repo_name = os.path.splitext(os.path.basename(repo_url))[0]
    local_path = os.path.join(base_dir, repo_name)

    if os.path.exists(local_path):
        print(f" Repo already exists at {local_path}, skipping clone.")
        return local_path

    os.makedirs(base_dir, exist_ok=True)
    print(f" Cloning {repo_url} → {local_path}")

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, repo_url, local_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f" Failed to clone repo: {e.stderr}")

    print(f" Repo cloned to {local_path}")
    return local_path


# --- Tool 1: Analyze Infra ---
@mcp.tool(name="analyze_infra", description="Analyze Terraform repo for right-sizing variables")
async def analyze_infra(params: AnalyzeInput, ctx: Context) -> AnalyzeOutput:
    """Clone GitHub repo (if needed) and analyze Terraform files."""
    repo_url = params.repo_url
    base_dir = params.base_dir or "C:\\repos"
    branch = params.branch or "main"

    # Step 1: Clone or reuse repo
    repo_path = await clone_repo_if_needed(repo_url, base_dir, branch)

    # Step 2: Analyze Terraform files
    tf_analysis = await asyncio.to_thread(analyze_tf_file, repo_path)

    # Step 3: Extract right-sizing vars
    rightsizing_info = await asyncio.to_thread(find_rightsizing_vars, tf_analysis)

    # Step 4: Save analysis JSON result
    output_file = os.path.join(repo_path, "tf_rightsizing_map.json")
    with open(output_file, "w") as f:
        json.dump(rightsizing_info, f, indent=2)

    return AnalyzeOutput(analysis_path=output_file, local_repo_path=repo_path)




@mcp.tool(name="clone_repo", description="Clone a GitHub repo to base_dir, keeping repo name as folder name")
def clone_repo(params: CloneInput, ctx: Context) -> CloneOutput:
    """Clone repo into base_dir or reuse existing clone."""
    os.makedirs(params.base_dir, exist_ok=True)

    # Derive folder name from repo URL (e.g., github.com/user/repo.git → repo)
    repo_name = os.path.splitext(os.path.basename(urlparse(params.repo_url).path))[0]
    clone_path = os.path.join(params.base_dir, repo_name)

    # If repo already exists locally, handle gracefully - reuse it (and optionally pull latest)
    if os.path.exists(clone_path):
        ctx.log(f" Repo already exists at {clone_path}. Skipping clone.")
        try:
            # Optional: Pull latest changes to stay up to date
            subprocess.run(["git", "-C", clone_path, "fetch", "--all"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
            subprocess.run(["git", "-C", clone_path, "reset", "--hard", f"origin/{params.branch}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
            ctx.log(f"Repo updated to latest branch '{params.branch}'.")
        except subprocess.CalledProcessError as e:
            ctx.log(f"Warning: Could not update repo: {e}")
        return CloneOutput(local_path=clone_path)

    #  Fresh clone if repo folder doesn’t exist
    ctx.log(f" Cloning {params.repo_url} into {clone_path} ...")
    subprocess.run(["git", "clone", "-b", params.branch, params.repo_url, clone_path], check=True,)

    ctx.log(f" Repo cloned successfully to {clone_path}")
    return CloneOutput(local_path=clone_path)
