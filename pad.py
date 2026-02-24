clean_json = llm_decision.strip()
if "```" in clean_json:
    clean_json = clean_json.replace("```json", "").replace("```", "")
parsed = json.loads(clean_json)


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

- example_1_file
- example_2_file

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


async def load_mcp_tools():
    async with Client(MCP_SERVER_URL) as client:
        discovery = await client.list_tools()
        tools = []

        for tool_def in discovery:

            def make_caller(tool_name):
                async def _caller(**kwargs):
                    async with Client(MCP_SERVER_URL) as client2:
                        result = await client2.call_tool(tool_name, kwargs)
                        return result.content
                return _caller

            caller_fn = make_caller(tool_def.name)

            tools.append(
                StructuredTool.from_function(
                    name=tool_def.name,
                    description=tool_def.description,
                    func=caller_fn,
                )
            )

        return tools



async def parent_executor(user_input: str):
    # Step 1: Router decides
    chosen_tool = await router.ainvoke({"input": user_input})
    chosen_tool = chosen_tool.strip()

    print("Router chose:", chosen_tool)

    # Step 2: Extract args
    extract_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
Extract JSON arguments required for the tool call.

Return ONLY a valid JSON object with:
- example_1_file
- example_2_file

User text:
{input}
"""
    )

    extractor = extract_prompt | llm | StrOutputParser()
    parsed_json = await extractor.ainvoke({"input": user_input})
    args = json.loads(parsed_json)

    # Step 3: Call correct subagent
    if chosen_tool == "yaml_updater_tool":
        return await yaml_agent.ainvoke(json.dumps(args))

    if chosen_tool == "locals_updater_tool":
        return await locals_agent.ainvoke(json.dumps(args))

    return f"Unknown tool: {chosen_tool}"

----

import asyncio
import streamlit as st

from langgraph.checkpoint.sqlite import AsyncSQLiteSaver
from langchain_core.messages import HumanMessage, AIMessage

# ======================
# CONFIG
# ======================
DB_PATH = "checkpoints.db"
THREAD_ID = "usr123"
N_HISTORY = 6

checkpointer = AsyncSQLiteSaver(DB_PATH)

# ======================
# CHECKPOINT HELPERS
# ======================
async def get_last_n_checkpoints(thread_id: str, n: int = 6):
    checkpoints = []
    async for ckpt in checkpointer.list(thread_id):
        checkpoints.append(ckpt)
    return checkpoints[-n:]


def extract_messages(checkpoints):
    messages = []
    for _, checkpoint, _ in checkpoints:
        messages.extend(
            checkpoint.get("channel_values", {}).get("messages", [])
        )
    return messages


def extract_mcp_logs(checkpoints):
    logs = []
    for _, checkpoint, _ in checkpoints:
        logs.extend(
            checkpoint.get("channel_values", {}).get("mcp_logs", [])
        )
    return logs


# ======================
# PLACEHOLDER ORCHESTRATOR
# Replace with your real one
# ======================
async def orchestrate(user_input: str, thread_id: str):
    # Example only — your LangGraph app.ainvoke() goes here
    pass


# ======================
# STREAMLIT APP
# ======================
st.set_page_config(page_title="LangGraph Chat", layout="wide")
st.title("LangGraph + SQLite Checkpoints")

# ----------------------
# Load history once
# ----------------------
if "history_loaded" not in st.session_state:
    checkpoints = asyncio.run(
        get_last_n_checkpoints(THREAD_ID, N_HISTORY)
    )
    st.session_state.messages = extract_messages(checkpoints)
    st.session_state.mcp_logs = extract_mcp_logs(checkpoints)
    st.session_state.history_loaded = True

# ----------------------
# Sidebar: MCP logs
# ----------------------
with st.sidebar:
    st.subheader("MCP Logs")
    for log in st.session_state.get("mcp_logs", []):
        st.code(log)

# ----------------------
# Chat history
# ----------------------
for msg in st.session_state.get("messages", []):
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ----------------------
# User input
# ----------------------
if user_input := st.chat_input("Ask something"):
    st.chat_message("user").write(user_input)

    # Run LangGraph
    asyncio.run(
        orchestrate(user_input=user_input, thread_id=THREAD_ID)
    )

    # Reload checkpoints after run
    checkpoints = asyncio.run(
        get_last_n_checkpoints(THREAD_ID, N_HISTORY)
    )

    st.session_state.messages = extract_messages(checkpoints)
    st.session_state.mcp_logs = extract_mcp_logs(checkpoints)

    st.rerun()






from langgraph.checkpoint.sqlite import AsyncSQLiteSaver

DB_PATH = "checkpoints.db"
checkpointer = AsyncSQLiteSaver(DB_PATH)


async def get_last_n_checkpoints(thread_id: str, n: int = 5):
    checkpoints = []
    async for ckpt in checkpointer.list(thread_id):
        checkpoints.append(ckpt)
    return checkpoints[-n:]


async def get_checkpoint_by_id(thread_id: str, checkpoint_id: str):
    async for config, checkpoint, metadata in checkpointer.list(thread_id):
        if config["configurable"].get("checkpoint_id") == checkpoint_id:
            return checkpoint
    return None



@app.get("/checkpoints", response_class=HTMLResponse)
async def list_checkpoints():
    checkpoints = await get_last_n_checkpoints(THREAD_ID, n=5)

    if not checkpoints:
        return "<h3>No checkpoints found</h3>"

    items = []
    for config, checkpoint, _ in reversed(checkpoints):
        ckpt_id = config["configurable"]["checkpoint_id"]
        ts = checkpoint.get("ts")
        workflow = checkpoint.get("route", {}).get("workflow", "unknown")

        items.append(
            f"""
            <li>
              <a href="/checkpoints/{ckpt_id}">
                {workflow} @ {ts}
              </a>
            </li>
            """
        )

    html = f"""
    <html>
      <body>
        <h2>Last 5 Checkpoints</h2>
        <ul>
          {''.join(items)}
        </ul>
      </body>
    </html>
    """

    return html


@app.get("/checkpoints/{checkpoint_id}")
async def checkpoint_detail(checkpoint_id: str):
    checkpoint = await get_checkpoint_by_id(THREAD_ID, checkpoint_id)

    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return JSONResponse(checkpoint)



import os
import hcl2
import json
from sentence_transformers import SentenceTransformer, util

# 1. Load the Semantic Model (Lightweight and fast for technical text)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_terraform_variables(root_dir):
    """Walks through a directory to extract variable names and descriptions."""
    var_library = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tf"):
                with open(os.path.join(root, file), 'r') as f:
                    try:
                        data = hcl2.load(f)
                        if 'variable' in data:
                            for var_block in data['variable']:
                                for name, attr in var_block.items():
                                    description = attr.get('description', [''])[0]
                                    # Create a rich text 'document' for the variable
                                    context = f"Variable: {name}. Description: {description}"
                                    var_library.append({
                                        "name": name,
                                        "context": context,
                                        "file": file
                                    })
                    except Exception as e:
                        print(f"Error parsing {file}: {e}")
    return var_library

def find_best_match(recommendation, var_library):
    """Matches a Turbonomic recommendation to the most similar TF variable."""
    # Convert library contexts to embeddings
    library_texts = [item['context'] for item in var_library]
    library_embeddings = model.encode(library_texts, convert_to_tensor=True)

    # Convert the Turbonomic recommendation into a searchable query
    # We include the entity name and the recommended action
    query = f"Target variable for {recommendation['entity_name']} to resize to {recommendation['new_value']}"
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(query_embedding, library_embeddings)[0]
    
    # Get the index of the highest score
    best_match_idx = cosine_scores.argmax().item()
    confidence = cosine_scores[best_match_idx].item()
    
    return var_library[best_match_idx], confidence

# --- EXECUTION ---

# Mock Turbonomic Data (This would normally come from an API or CSV)
turbo_json = {
    "entity_name": "payment-processor-vm",
    "current_value": "Standard_D2s_v3",
    "new_value": "Standard_B2s",
    "resource_group": "production-rg"
}

# Path to your Terraform repositories
tf_dir = "./my-terraform-repo"

print("Scanning Terraform files...")
vars_found = extract_terraform_variables(tf_dir)

if not vars_found:
    print("No variables found in the directory.")
else:
    print(f"Found {len(vars_found)} variables. Finding best match...")
    match, score = find_best_match(turbo_json, vars_found)

    print("-" * 30)
    print(f"TURBONOMIC ENTITY: {turbo_json['entity_name']}")
    print(f"RECOMMENDED MATCH: {match['name']}")
    print(f"CONFIDENCE SCORE:  {score:.4f}")
    print(f"SOURCE FILE:      {match['file']}")
    print("-" * 30)

    if score < 0.75:
        print("WARNING: Low confidence. Manual verification recommended.")

import ssl
import os

# This tells Python to ignore SSL certificate errors for this execution
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



import json
from difflib import SequenceMatcher

def get_similarity(a, b):
    """Calculates text similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def load_json_file(file_path):
    """Safely loads a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return None

def match_files(terraform_file, finops_file, output_file="matched_results.json"):
    # Load the data
    tf_data = load_json_file(terraform_file)
    finops_data = load_json_file(finops_file)

    if tf_data is None or finops_data is None:
        return

    sizing_vars = tf_data.get("sizing_variables", [])
    results = []

    print(f"Processing {len(finops_data)} FinOps items against {len(sizing_vars)} Terraform variables...")

    for finops in finops_data:
        best_match = None
        highest_score = -1

        for tf in sizing_vars:
            # 1. Environment Filter (Highest Importance)
            # Validates that we are looking at the right stack (dev/prod/qa)
            env_match = 1.0 if tf.get("environment") == finops.get("environment_name") else 0.0
            
            # 2. Semantic Match: id -> variable_name
            name_score = get_similarity(tf.get("id"), finops.get("variable_name"))
            
            # 3. Semantic Match: category -> variable_description
            desc_score = get_similarity(tf.get("category"), finops.get("variable_description"))
            
            # 4. Value Match: value -> variable_value_old
            # We convert to string to handle both int and float comparisons
            val_match = 1.0 if str(tf.get("value")) == str(finops.get("variable_value_old")) else 0.0

            # Weighted Scoring Calculation
            # Weighting: Env(40%), Name Similarity(30%), Description(20%), Old Value(10%)
            total_score = (env_match * 0.4) + (name_score * 0.3) + (desc_score * 0.2) + (val_match * 0.1)

            if total_score > highest_score:
                highest_score = total_score
                best_match = tf

        # Append the successful match
        if best_match:
            results.append({
                "finops_input": {
                    "variable": finops.get("variable_name"),
                    "env": finops.get("environment_name")
                },
                "terraform_match": {
                    "id": best_match.get("id"),
                    "path": best_match.get("terraform_path"),
                    "current_value": best_match.get("value")
                },
                "confidence": round(highest_score, 4)
            })

    # Save results to a file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table to console
    print(f"\n{'FinOps Var':<15} | {'Match ID':<15} | {'Score':<8} | {'Terraform Path'}")
    print("-" * 100)
    for res in results:
        print(f"{res['finops_input']['variable']:<15} | "
              f"{res['terraform_match']['id']:<15} | "
              f"{res['confidence']:<8} | "
              f"{res['terraform_match']['path']}")

    print(f"\nFull matching report saved to: {output_file}")

if __name__ == "__main__":
    # Specify your file names here
    match_files("terraform_analysis.json", "finops_data.json")




"""
Main orchestrator.
Uses langchain, langgraph and LLMto orchestrate the whole workflow in one process.

Currently under construction
Old version of code that must be replaced
"""

# import traceback
from pathlib import Path
import traceback
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Any, Dict, List, Union, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from astra_azure_openai_client import AsyncAzureOpenAIClient
from workflows.rightsize_graph import build_rightsize_graph, RightSizeState
from utils import (
    serialize_result,
    safe_parse_llm_output,
    extract_emails_from_cmdb_metadata,
    extract_finops_data_from_toolresult,
    notify_stakeholders,
)
from config import APP_CFG
from chatbot_logger import logger
from astra_mcp_utils import call_mcp_tool, McpFastAPIClient, list_mcp_tools
from db_utils import save_workflow_state, load_workflow_state, del_workflow_state
from database import get_app_db

# Set CONSTANTS and variables
MCP_TOOLS_URL = APP_CFG.MCP_TOOLS_URL
MCP_SERVER_URL = APP_CFG.MCP_SERVER_URL
MCP_AGENT_URL = APP_CFG.MCP_AGENT_URL
MCP_AGENT_PORT = APP_CFG.MCP_AGENT_PORT
CLONE_DIR = Path(APP_CFG.CLONE_DIR)
ANALYSIS_OUT_DIR = Path(APP_CFG.ANALYSIS_OUT_DIR)

# Async Azure OpenAI client  # TODO: improve config so only AZ part is used here
llm_client = AsyncAzureOpenAIClient(az_cfg=APP_CFG)


# Langgraph setup - state definition, implement nodes
class OrchestratorState(BaseModel):
    """Orchestration state"""

    user_input: str
    session_id: Optional[str] = None
    route: dict = None
    messages: list = []
    mcp_logs: list = []
    final_answer: Any | None = None
    user_id: Optional[str] = None  # TODO: make this mandatory later
    branch: Optional[str] = None
    rightsize_graph: Any = None  # Pre-built workflow graph
    param_collection_attempts: int = 0  # Tracks how many times we have asked for missing params


# Implement workflow - update_rightsize_workflow
async def run_update_rightsize_workflow(
    # repo_url: str, env: str, app_id: str="APP0002202", branch: str="main"
    repo_url: str,
    env: str,
    app_id: str,
    user_id: str,
    session_id: str = None,
    branch: str = "main",
    graph=None,  # Pre-built graph from FastAPI app.state
) -> RightSizeState:
    """
    Run the Astra MCP Agent workflow for updating rightsize.

    Args:
        repo_url (str): The repository URL.
        env (str): The environment name.
        app_id (str): The application ID.
        session_id (str): The session ID for concurrent workflow execution.
        branch (str): The git branch name.

    Returns:
        RightSizeState: The final state after workflow execution.
    May be create and add to state feature branch name already now
    """
    params = {
        "repo_url": repo_url,
        "environment": env,
        "app_id": app_id,
        "user_id": user_id,
        "session_id": session_id,
        "branch": branch,
        "local_base_dir": CLONE_DIR,
        "aanalysis_base_dir": ANALYSIS_OUT_DIR,
    }
    logger.debug(f"Starting workflow with parameters: {params}")

    # Validate that graph is provided (should be from FastAPI app.state)
    if graph is None:
        raise ValueError(
            "Workflow graph not provided. "
            "Graph must be initialized at FastAPI startup via lifespan context manager."
        )

    # acquire app_db session
    app_db = next(get_app_db())

    # Use pre-built graph from FastAPI app.state (no need to build here)
    update_rightsize_workflow_graph = graph
    logger.debug("Using pre-built workflow graph from FastAPI app.state")

    # Initialize the state object separately
    try:
        initial_state = RightSizeState(**params)
        await save_workflow_state(
            user_id=user_id, session_id=session_id, state=initial_state, app_db=app_db
        )
        logger.debug("Workflow state initialized and saved to DB.")
    except Exception as e:
        logger.error(f"Failed to initialize state: {e}\n{traceback.format_exc()}")
        raise

    # Invoke the graph with the state at runtime
    try:
        logger.debug(f"About to invoke workflow graph with: {initial_state = }")
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        logger.debug(f"Invoking workflow with checkpoint config: {config}")
        result_state = await update_rightsize_workflow_graph.ainvoke(
            initial_state, config=config
        )

        # Convert dict result to RightSizeState object
        # LangGraph ainvoke returns dict, not Pydantic model
        result_state_obj = RightSizeState(**result_state)

        # Save updated state to WorkflowStateTable (includes HITL pause state)
        # This allows HITL endpoints to find the session and check hitl_pending flag
        await save_workflow_state(
            user_id=user_id,
            session_id=session_id,
            state=result_state_obj,
            app_db=app_db,
        )
        logger.debug(
            f"Workflow state saved after execution (hitl_pending={result_state_obj.hitl_pending})"
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}\n{traceback.format_exc()}")
        raise

    return result_state_obj


def route_node(state: OrchestratorState):
    """Node to route the workflow based on the user input."""
    return state.route["workflow"]


async def general_question_node(state: OrchestratorState):
    """Node to answer general questions."""
    answer = state.route["answer"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def single_tool_node(state: OrchestratorState):
    """Node to call a single tool."""
    tool = state.route["tool"]
    params = state.route["params"]
    logger.debug(f"Inside single_tool_node...Input....: {tool = }, {params = }")
    result = await call_mcp_tool(MCP_SERVER_URL, tool, params)

    new_mcp_logs = state.mcp_logs + [
        {"tool": tool, "params": params, "status": "Success"}
    ]
    new_messages = state.messages + [
        {
            "role": "assistant",
            "content": (
                f"Tool `{tool}` executed with params {params}. Result:\n{result}"
            ),
        },
    ]
    # , {"role": "user", "content": result}]

    logger.debug(f"Inside single_tool_node...Result...: {result}")

    return {"final_answer": result, "mcp_logs": new_mcp_logs, "messages": new_messages}


async def workflow_node(state: OrchestratorState):
    """Node to orchestrate main Astra workflow.

    Handles two entry scenarios:
      1. Direct invocation - route["params"] already has repo_url, appid, environment.
      2. Chained from greeting_node - params may be empty; prompt user for required details.
    """
    if state.route.get("workflow") != "update_rightsize_repo":
        return {"final_answer": "Unsupported workflow"}
    logger.debug(f"Inside workflow_node. OrchestratorState: {state = }")

    params = state.route.get("params", {})
    attempts = state.param_collection_attempts

    # Determine which required params are still missing
    missing = []
    if not params.get("repo_url"):
        missing.append("**Repository URL** (e.g. https://github.com/org/repo)")
    if not params.get("appid"):
        missing.append("**App ID** (e.g. APP0002202)")
    if not params.get("environment"):
        missing.append("**Environment** (e.g. dev, qa, prod)")

    if missing:
        if attempts == 0:
            # First ask - request all 3 params clearly
            prompt_msg = (
                "To start the rightsizing workflow I need a few details:\n\n"
                "- **Repository URL** (e.g. https://github.com/org/repo)\n"
                "- **App ID** (e.g. APP0002202)\n"
                "- **Environment** (e.g. dev, qa, prod)\n\n"
                "Please provide all three and I will kick off the workflow right away."
            )
            logger.info("Attempt 1: prompting user for all required params")
        elif attempts == 1:
            # Second ask - tell user exactly what is still missing
            missing_list = "\n".join(f"- {m}" for m in missing)
            prompt_msg = (
                f"I still need the following to proceed:\n\n{missing_list}\n\n"
                "Could you please provide these?"
            )
            logger.info(f"Attempt 2: prompting user for still-missing params: {missing}")
        else:
            # Exceeded 2 attempts - give up gracefully
            prompt_msg = (
                "I was unable to collect the required details after two attempts. "
                "Please restart and provide the Repository URL, App ID, and Environment "
                "to run the rightsizing workflow."
            )
            logger.info("Exceeded 2 param collection attempts - aborting")
            return {
                "final_answer": prompt_msg,
                "messages": state.messages + [{"role": "assistant", "content": prompt_msg}],
                "param_collection_attempts": attempts + 1,
            }

        return {
            "final_answer": prompt_msg,
            "messages": state.messages + [{"role": "assistant", "content": prompt_msg}],
            "param_collection_attempts": attempts + 1,
        }

    repo_url = params["repo_url"]
    app_id = params["appid"]
    env = params["environment"]
    session_id = state.session_id or "s001"
    user_id = state.user_id
    branch = state.branch or "main"

    # Get pre-built workflow graph from state (passed down from FastAPI app)
    graph = state.rightsize_graph
    logger.debug(f"workflow_node: Retrieved graph from state: {graph}")

    if graph is None:
        raise ValueError(
            "Workflow graph not initialized. "
            "Check FastAPI lifespan startup logs for checkpointer initialization errors."
        )

    result = await run_update_rightsize_workflow(
        repo_url=repo_url,
        env=env,
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        branch=branch,
        graph=graph,  # Pass pre-built graph
    )

    # Create user-friendly message instead of dumping full state
    # Extract progress messages from workflow execution
    progress_messages = []
    if result.messages:
        # Filter for assistant messages (progress updates from workflow nodes)
        for msg in result.messages:
            if isinstance(msg, dict):
                if msg.get("type") == "ai" or msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and not content.startswith("⏸"):  # Skip pause messages
                        progress_messages.append(content)

    if result.hitl_checkpoint:
        # Workflow paused at HITL checkpoint
        checkpoint_name = result.hitl_checkpoint
        friendly_message = f"⏸ **Workflow Paused - Approval Required**\n\n"

        # Add progress summary
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Checkpoint:** {checkpoint_name}\n\n"

        if checkpoint_name == "finops_validation":
            rec_count = len(result.recommendations) if result.recommendations else 0
            friendly_message += (
                f"📊 **Status:** Fetched {rec_count} rightsizing recommendation(s) from FinOps\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "analysis_approval":
            var_count = (
                result.analysis_output.get("sizing_variables_count", 0)
                if result.analysis_output
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Analyzed Terraform infrastructure - found {var_count} sizing variable(s)\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "commit_approval":
            files_modified = (
                len(result.hitl_data.get("files_modified", []))
                if result.hitl_data
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Infrastructure updates ready for commit\n"
                f"📄 Modified {files_modified} file(s)\n"
                f"🌿 Branch: {result.feature_branch}\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )

        final_answer = friendly_message
    elif result.error:
        # Workflow failed
        friendly_message = "❌ **Workflow Failed**\n\n"

        # Show progress before failure
        if progress_messages:
            friendly_message += "**Progress before failure:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Error:** {result.error}"
        final_answer = friendly_message
    elif result.pr:
        # Workflow completed successfully
        friendly_message = "✅ **Workflow Complete!**\n\n"

        # Show all progress steps
        if progress_messages:
            friendly_message += "**Completed steps:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**Pull Request:** {result.pr}\n"
            f"**Branch:** {result.feature_branch}\n\n"
            f"👉 **Next Step:** Review and merge the pull request."
        )
        final_answer = friendly_message

    else:
        # Workflow still running or completed without PR
        friendly_message = f"✅ **Rightsizing Workflow Executed**\n\n"

        # Show progress
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**App ID:** {app_id}\n"
            f"**Environment:** {env}\n"
            f"**Session ID:** {session_id}\n\n"
            f"👉 Check workflow status in the sidebar."
        )
        final_answer = friendly_message

    new_mcp_logs = state.mcp_logs + [
        {
            "tool": "update_rightsize_workflow",
            "params": {
                "repo_url": repo_url,
                "environment": env,
                "app_id": app_id,
                "session_id": session_id,
            },
            "status": "Success",
        }
    ]
    new_messages = state.messages + [{"role": "assistant", "content": final_answer}]

    return {
        "final_answer": final_answer,
        "mcp_logs": new_mcp_logs,
        "messages": new_messages,
    }


async def none_node(state: OrchestratorState):
    """Node to handle empty input."""
    answer = state.route["message"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


# async def greeting_node(state: OrchestratorState):
#     """Node to handle greeting input."""
#     answer = state.route["message"]
#     new_messages = state.messages + [{"role": "assistant", "content": answer}]
#     return {"final_answer": answer, "messages": new_messages}


async def greeting_node(state: OrchestratorState):
    """Node to handle greeting input, respond with greeting, then immediately trigger workflow_node.

    Flow:
      1. Send a friendly greeting message back to the user.
      2. Always chain into workflow_node so the conversation continues naturally
         into the update_rightsize_repo workflow without requiring a second user message.
    """
    greeting_text = state.route.get("message", "Hi there! I can help you automate FinOps and Infra updates.")
    logger.debug(f"Inside greeting_node. Sending greeting and chaining to workflow_node.")

    # Append greeting to conversation history
    new_messages = state.messages + [{"role": "assistant", "content": greeting_text}]

    # Build updated state that carries the greeting in messages and routes to the workflow.
    # Override route so workflow_node knows which workflow to run.
    updated_route = {
        **state.route,
        "workflow": "update_rightsize_repo",
        # params may be absent on a bare greeting - workflow_node will prompt for them
        "params": state.route.get("params", {}),
    }
    updated_state = state.model_copy(update={
        "messages": new_messages,
        "final_answer": greeting_text,
        "route": updated_route,
    })

    # Chain directly into workflow_node - natural conversation continues
    workflow_result = await workflow_node(updated_state)

    # Prepend the greeting to whatever workflow_node returned so the user sees both
    workflow_answer = workflow_result.get("final_answer", "")
    combined_answer = f"{greeting_text}\n\n{workflow_answer}" if workflow_answer else greeting_text

    # Merge messages: greeting already in new_messages; workflow messages may overlap, use workflow's list
    final_messages = workflow_result.get("messages", new_messages)

    return {
        "final_answer": combined_answer,
        "messages": final_messages,
        "mcp_logs": workflow_result.get("mcp_logs", state.mcp_logs),
    }


# After greeting, if workflow is requested, flow to workflow_node, else end
def greetings_edge_selector(state: OrchestratorState):
    if state.route.get("workflow") == "update_rightsize_repo":
        return "update_rightsize_repo"
    return END


async def orchestrator(
    user_input: str,
    mcp_logs: list,
    messages: list,
    session_id: str,
    user_id: str,
    rightsize_graph=None,  # Pre-built graph passed from FastAPI app.state
):
    """
    Main orchestrator of the Astra MCP Agent Server.
    Organizes:
        - different types of workflows using LangGraph router pattern
        - external communication using /chat and /tools endpoints.
    """
    # Ensure session_id is not None; should fail later if not provided
    session_id = session_id or str(uuid4())

    # Ensure user_id is not None; should fail later if not provided
    user_id = user_id or "fake_user"

    # Add user's current input to messages BEFORE processing
    # This ensures the user's message is included in the conversation history
    messages_with_input = messages + [{"role": "user", "content": user_input}]

    # Call LLM router
    logger.debug(
        f"In the orchestrator, calling LLM with: {user_input = }, {messages = }"
    )
    if user_input.lower() in ["hi", "hello", "hey"]:
        # Route to greetings node; greeting_node will chain into workflow_node automatically.
        # No params yet - workflow_node will prompt the user for missing details (env, repo, etc.)
        route_decision = {
            "workflow": "greetings",
            "message": "Hi there! I can help you automate FinOps and Infra updates.",
            "params": {},  # Empty params - workflow_node will ask user for them
        }
    else:
        route_decision = await llm_client.ask_llm(user_input, messages)
    # route_decision = await llm_client.ask_llm(user_input, messages)
    # route = await llm_client.ask_llm_auto_discovery(user_input, messages)  # Uncomment for auto-discovery. TODO: fix pa
    logger.debug(f"Route: {route_decision = }")

    # Create graph state with accumulated messages (including current user input)
    orch_state = OrchestratorState(
        user_input=user_input,
        session_id=session_id,
        messages=messages_with_input,  # Include user's current message
        mcp_logs=mcp_logs,
        route=route_decision,
        final_answer=None,
        user_id=user_id,
        rightsize_graph=rightsize_graph,  # Pass pre-built graph through state
    )
    logger.debug(f"Initializing with OrchestratorState: {orch_state = }")

    # Build LangGraph
    orchestrator_graph = StateGraph(OrchestratorState)
    orchestrator_graph.add_node("router", lambda s: {"route": s.route})
    orchestrator_graph.set_entry_point("router")

    orchestrator_graph.add_edge("router", orch_state.route["workflow"])

    orchestrator_graph.add_node("none", none_node)
    orchestrator_graph.add_node("greetings", greeting_node)
    orchestrator_graph.add_node("general_question", general_question_node)
    orchestrator_graph.add_node("single_tool", single_tool_node)
    orchestrator_graph.add_node("update_rightsize_repo", workflow_node)

    orchestrator_graph.add_edge("none", END)
    orchestrator_graph.add_edge("greetings", END)
    # orchestrator_graph.add_edge("greetings", greetings_edge_selector)
    orchestrator_graph.add_edge("general_question", END)
    orchestrator_graph.add_edge("single_tool", END)
    orchestrator_graph.add_edge("update_rightsize_repo", END)

    logger.debug(f"Orchestrator Graph: {orchestrator_graph}")

    # Compile and execute with no checkpointer
    app = orchestrator_graph.compile()
    logger.info("Orchestrator compiled.")
    logger.debug(f"App: {app} will be invoked with {orch_state = }")
    try:
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id}}
        logger.debug(f"Invoking orchestrator with checkpoint config: {config}")
        result_state = await app.ainvoke(orch_state, config=config)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e} {traceback.format_exc()}")
        result_state = {"final_answer": f"Orchestrator failed: {e}"}
    logger.debug(f"In orchestrator result State before return: {result_state}")
    logger.info("Orchestration complete.")

    return (
        result_state["final_answer"],
        result_state["messages"],
        result_state["mcp_logs"],
    )




import os
import sys
import asyncio
import httpx
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from chat_ui_logger import logger

# ---- Configuration ----
load_dotenv()
LOCAL_DEPLOYMENT = not (len(sys.argv) > 1 and sys.argv[1] == "dev")
CHAT_ENDPOINT = (
    os.getenv("ASTRA_AGENT_CHAT_ENDPOINT", "http://localhost:8001/astrachatbotui/chat")
    if LOCAL_DEPLOYMENT else
    "https://astrachatbot-dev-eastus2.humana.com/astrachatbotui/chat"
)

agent_token = os.getenv("ASTRA_API_TOKEN")

# Greeting triggers: when user sends one of these, we send a greeting call
# immediately followed by param collection so conversation flows naturally.
GREETING_TRIGGERS = {"hi", "hello", "hey", "hi!", "hello!", "hey!"}

# Workflow triggers: skip greeting entirely and jump straight into param collection.
WORKFLOW_TRIGGERS = {"go update", "start update"}

# ---- Helper Functions ----


def _ask_for_next_missing_param() -> str:
    """Return the next question to ask based on what params are still missing.
    Returns None if all 3 params are collected.
    """
    params = st.session_state.workflow_params
    if not params.get("appid"):
        st.session_state.pending_param = "appid"
        return "What is the **App ID** for this rightsizing operation? (e.g. APP0002202)"
    if not params.get("repo_url"):
        st.session_state.pending_param = "repo_url"
        return "What is the **Repository URL**? (e.g. https://github.com/org/repo)"
    if not params.get("environment"):
        st.session_state.pending_param = "environment"
        return "Which **environment** would you like to update? (e.g. dev, qa, prod)"
    return None  # All params collected


def _store_user_answer(user_text: str):
    """Store the user's latest answer into workflow_params based on pending_param."""
    pending = st.session_state.pending_param
    if pending and user_text.strip():
        st.session_state.workflow_params[pending] = user_text.strip()
        st.session_state.pending_param = None


def _build_workflow_trigger() -> str:
    """Build the workflow trigger string from collected params."""
    p = st.session_state.workflow_params
    return (
        f"update the right-size variables for appid {p['appid']} "
        f"in repository {p['repo_url']} "
        f"for environment {p['environment']}"
    )


def _reset_param_collection():
    """Reset all param collection state."""
    st.session_state.collecting_params = False
    st.session_state.workflow_params = {}
    st.session_state.param_attempts = 0
    st.session_state.pending_param = None

async def call_chat_api(
    mode: str,
    user_id: str,
    agent_token: str,
    session_id=None,
    **kwargs
):
    """Call unified /chat endpoint. If session_id is None, backend will assign one."""
    payload = {
        "mode": mode,
        "user_id": user_id,
        **kwargs
    }
    if session_id is not None:
        payload["session_id"] = session_id

    headers = {
        "Authorization": f"Bearer {agent_token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=600.0, verify=False) as client:
        response = await client.post(CHAT_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def get_workflow_stage_message(prompt: str, checkpoint: str = None) -> str:
    """Get workflow stage message based on prompt and checkpoint."""
    if checkpoint:
        return {
            'finops_validation': "📊 Validating FinOps recommendations with CMDB...",
            'analysis_approval': "🔍 Analyzing Terraform infrastructure patterns...",
            'commit_approval': "📋 Preparing commit and pull request..."
        }.get(checkpoint, "⏳ Processing workflow stage...")
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in ['clone', 'repository', 'repo']):
        return "📦 Cloning repository..."
    if any(kw in prompt_lower for kw in ['analyze', 'analysis', 'terraform']):
        return "🔍 Analyzing infrastructure..."
    if any(kw in prompt_lower for kw in ['update', 'modify', 'rightsize', 'resize']):
        return "✏️ Updating configuration..."
    if any(kw in prompt_lower for kw in ['commit', 'push', 'pr']):
        return "📋 Creating pull request..."
    if any(kw in prompt_lower for kw in ['recommend', 'finops']):
        return "💡 Fetching recommendations..."
    return "🚀 Processing request..."


def initialize_session_state():
    """Initialize session state with fake values."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = "fake_user"  # Default username
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # No session yet; backend will assign
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_raw_response" not in st.session_state:
        st.session_state.show_raw_response = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "collecting_params" not in st.session_state:
        st.session_state.collecting_params = False  # True while gathering workflow params after greeting
    if "workflow_params" not in st.session_state:
        st.session_state.workflow_params = {}  # Collected so far: appid, repo_url, environment
    if "param_attempts" not in st.session_state:
        st.session_state.param_attempts = 0  # How many collection rounds attempted
    if "pending_param" not in st.session_state:
        st.session_state.pending_param = None  # Which param we are currently asking for
    agent_token_input = st.text_input("Enter agent token:", type="password")
    if agent_token_input:
        st.session_state.agent_token = agent_token_input
    # if "agent_token" not in st.session_state:
    #     st.session_state.agent_token = agent_token


def _display_field_value(field, value):
    """Helper to display a field in the sidebar"""
    if isinstance(value, (dict, list)):
        with st.expander(field, expanded=False):
            st.json(value)
    else:
        st.write(f"**{field}:** `{value}`")


def display_result_summary(result):
    """Show all top-level fields except those displayed elsewhere"""
    if not isinstance(result, dict):
        return
    ignore_keys = {"result", "history", "mcp_logs"}
    fields = [k for k in result if k not in ignore_keys]
    if not fields:
        st.info("No extra fields in API response.")
        return
    for field in fields:
        value = result[field]
        _display_field_value(field, value)


# ---- Page Configuration ----

st.set_page_config(
    page_title="Astra Rightsizing Workflow",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Session State Initialization ----
initialize_session_state()

# ---- Sidebar ----

with st.sidebar:
    st.header("🖥️ Astra Rightsizing Workflow")

    # User controls for identity and session
    st.subheader("🔑 User & Session")

    # Username entry
    user_input = st.text_input(
        "Username:",
        value=st.session_state.user_id,
        key="user_id_input"
    )
    if user_input and user_input != st.session_state.user_id:
        st.session_state.user_id = user_input
        st.success(f"Username set to: {user_input}")
        st.rerun()

    # Session entry
    session_input = st.text_input(
        "Session ID (optional, for existing chat):",
        value=st.session_state.session_id or "",
        key="session_id_input"
    )
    if session_input and session_input != (st.session_state.session_id or ""):
        st.session_state.session_id = session_input
        st.session_state.messages = []
        st.success(f"Connected to session: {session_input}")
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.success("A new session will start on your next message.")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("✅ Chat cleared")
            st.rerun()

    st.divider()

    # Diagnostics
    with st.expander("🔧 Diagnostic Tools", expanded=False):
        st.session_state.show_raw_response = st.checkbox(
            "Show Raw API Responses",
            value=st.session_state.show_raw_response
        )
        st.caption(f"**Backend:** {CHAT_ENDPOINT}")
        st.caption(f"**Mode:** {'Local' if LOCAL_DEPLOYMENT else 'DEV'}")

    # Session summary (read-only display, if active)
    if st.session_state.session_id:
        st.caption(f"**Active Session:** `{st.session_state.session_id}`")
    st.caption(f"**User:** `{st.session_state.user_id}`")

    # ---- Last API Response Fields (NEW FEATURE) ----
    if st.session_state.last_result:
        st.divider()
        st.subheader("🌐 Last API Response Summary")
        display_result_summary(st.session_state.last_result)
    else:
        st.info("No API response yet.")

# ---- Main Chat Interface ----

st.title("💬 Astra Rightsizing Workflow")
st.caption("Infrastructure optimization powered by AI")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
prompt = st.chat_input("Say 'hi' to start or type correct command "
"- e.g., 'update rightsize variables for app APP123 with repo url "
"https://github.com/username/repo.git environment dev'")

# ── Param collection handler ─────────────────────────────────────────────────
# When collecting_params is True, we are in a guided param-collection dialogue.
# Each user reply is captured here, stored, and the next question is asked.
# Once all 3 params are collected (or 3 attempts exhausted), Call 2 fires.
MAX_PARAM_ATTEMPTS = 3

if st.session_state.collecting_params and prompt:
    # Store this answer for whatever param we were waiting on
    _store_user_answer(prompt)
    st.session_state.param_attempts += 1

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if all params are now collected
    next_question = _ask_for_next_missing_param()

    if next_question is None or st.session_state.param_attempts >= MAX_PARAM_ATTEMPTS:
        # All params collected (or max attempts reached) - fire workflow
        params = st.session_state.workflow_params
        all_collected = all(params.get(k) for k in ("appid", "repo_url", "environment"))

        if all_collected:
            workflow_trigger = _build_workflow_trigger()
            with st.chat_message("assistant"):
                with st.spinner("🚀 Kicking off the rightsizing workflow..."):
                    try:
                        workflow_result = asyncio.run(call_chat_api(
                            mode="chat",
                            user_id=st.session_state.user_id,
                            agent_token=st.session_state.agent_token,
                            session_id=st.session_state.session_id,
                            user_input=workflow_trigger,
                            history=st.session_state.messages,
                            mcp_logs=[]
                        ))
                        workflow_reply = workflow_result.get("result", "")
                        if workflow_reply:
                            st.write(workflow_reply)
                        if st.session_state.show_raw_response:
                            with st.expander("🔍 Raw API Response (workflow trigger)"):
                                st.json(workflow_result)
                        wf_session_id = workflow_result.get("session_id")
                        if wf_session_id and st.session_state.session_id != wf_session_id:
                            st.session_state.session_id = wf_session_id
                        wf_history = workflow_result.get("history", [])
                        if wf_history:
                            st.session_state.messages = wf_history
                        elif workflow_reply:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": workflow_reply}
                            )
                        st.session_state.last_result = workflow_result
                    except Exception as e:
                        error_msg = f"❌ Error triggering workflow: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
        else:
            # Max attempts reached but still missing params - give up gracefully
            give_up_msg = (
                "I was unable to collect all required details after several attempts. "
                "Please try again and provide the App ID, Repository URL, and Environment."
            )
            with st.chat_message("assistant"):
                st.write(give_up_msg)
            st.session_state.messages.append({"role": "assistant", "content": give_up_msg})

        _reset_param_collection()
        asyncio.run(asyncio.sleep(0.3))
        st.rerun()

    else:
        # Still missing params - ask next question
        with st.chat_message("assistant"):
            st.write(next_question)
        st.session_state.messages.append({"role": "assistant", "content": next_question})
        asyncio.run(asyncio.sleep(0.3))
        st.rerun()

elif prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Detect prompt type:
    #   is_greeting      → Call 1 gets greeting reply, then param collection starts
    #   is_workflow_trigger → skip greeting entirely, jump straight into param collection
    is_greeting = prompt.strip().lower() in GREETING_TRIGGERS
    is_workflow_trigger = prompt.strip().lower() in WORKFLOW_TRIGGERS

    # Workflow trigger: skip greeting, go straight into param collection
    if is_workflow_trigger:
        st.session_state.collecting_params = True
        st.session_state.workflow_params = {}
        st.session_state.param_attempts = 0
        st.session_state.pending_param = None
        first_question = _ask_for_next_missing_param()
        with st.chat_message("assistant"):
            st.write(first_question)
        st.session_state.messages.append({"role": "assistant", "content": first_question})
        asyncio.run(asyncio.sleep(0.3))
        st.rerun()

    # Call API with session_id (can be None on first call)
    with st.chat_message("assistant"):
        spinner_msg = get_workflow_stage_message(prompt)
        with st.spinner(spinner_msg):
            try:
                # ── Call 1: send the user's message (greeting or normal) ──────────
                result = asyncio.run(call_chat_api(
                    mode="chat",
                    user_id=st.session_state.user_id,
                    agent_token=st.session_state.agent_token,
                    session_id=st.session_state.session_id,
                    user_input=prompt,
                    history=st.session_state.messages[:-1],
                    mcp_logs=[]
                ))

                reply = result.get("result", "No response received")
                st.write(reply)

                # Show raw response in diagnostic mode
                if st.session_state.show_raw_response:
                    with st.expander("🔍 Raw API Response (greeting)"):
                        st.json(result)

                # Update session_id from backend if it's new
                backend_session_id = result.get("session_id")
                if backend_session_id and st.session_state.session_id != backend_session_id:
                    st.session_state.session_id = backend_session_id

                # Update history from backend if available
                returned_history = result.get("history", [])
                if returned_history:
                    st.session_state.messages = returned_history
                else:
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                # Store last result for sidebar summary
                st.session_state.last_result = result

                # ── Call 2 (greeting only): collect params then trigger workflow_node ──
                # After greeting, we enter a param-collection loop (up to 3 attempts).
                # Each Streamlit rerun advances the loop by one param question/answer.
                # Once all 3 params are collected, we fire the real workflow trigger.
                if is_greeting and not is_workflow_trigger:
                    # Start param collection mode
                    st.session_state.collecting_params = True
                    st.session_state.workflow_params = {}
                    st.session_state.param_attempts = 0
                    st.session_state.pending_param = None
                    # Ask for the first missing param immediately
                    first_question = _ask_for_next_missing_param()
                    st.write(first_question)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": first_question}
                    )

                asyncio.run(asyncio.sleep(0.5))
                st.rerun()

            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})







"""
Main orchestrator.
Uses langchain, langgraph and LLMto orchestrate the whole workflow in one process.

Currently under construction
Old version of code that must be replaced
"""

# import traceback
from pathlib import Path
import traceback
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Any, Dict, List, Union, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from astra_azure_openai_client import AsyncAzureOpenAIClient
from workflows.rightsize_graph import build_rightsize_graph, RightSizeState
from utils import (
    serialize_result,
    safe_parse_llm_output,
    extract_emails_from_cmdb_metadata,
    extract_finops_data_from_toolresult,
    notify_stakeholders,
)
from config import APP_CFG
from chatbot_logger import logger
from astra_mcp_utils import call_mcp_tool, McpFastAPIClient, list_mcp_tools
from db_utils import save_workflow_state, load_workflow_state, del_workflow_state
from database import get_app_db

# Set CONSTANTS and variables
MCP_TOOLS_URL = APP_CFG.MCP_TOOLS_URL
MCP_SERVER_URL = APP_CFG.MCP_SERVER_URL
MCP_AGENT_URL = APP_CFG.MCP_AGENT_URL
MCP_AGENT_PORT = APP_CFG.MCP_AGENT_PORT
CLONE_DIR = Path(APP_CFG.CLONE_DIR)
ANALYSIS_OUT_DIR = Path(APP_CFG.ANALYSIS_OUT_DIR)

# Async Azure OpenAI client  # TODO: improve config so only AZ part is used here
llm_client = AsyncAzureOpenAIClient(az_cfg=APP_CFG)


# Langgraph setup - state definition, implement nodes
class OrchestratorState(BaseModel):
    """Orchestration state"""

    user_input: str
    session_id: Optional[str] = None
    route: dict = None
    messages: list = []
    mcp_logs: list = []
    final_answer: Any | None = None
    user_id: Optional[str] = None  # TODO: make this mandatory later
    branch: Optional[str] = None
    rightsize_graph: Any = None  # Pre-built workflow graph


# Implement workflow - update_rightsize_workflow
async def run_update_rightsize_workflow(
    # repo_url: str, env: str, app_id: str="APP0002202", branch: str="main"
    repo_url: str,
    env: str,
    app_id: str,
    user_id: str,
    session_id: str = None,
    branch: str = "main",
    graph=None,  # Pre-built graph from FastAPI app.state
) -> RightSizeState:
    """
    Run the Astra MCP Agent workflow for updating rightsize.

    Args:
        repo_url (str): The repository URL.
        env (str): The environment name.
        app_id (str): The application ID.
        session_id (str): The session ID for concurrent workflow execution.
        branch (str): The git branch name.

    Returns:
        RightSizeState: The final state after workflow execution.
    May be create and add to state feature branch name already now
    """
    params = {
        "repo_url": repo_url,
        "environment": env,
        "app_id": app_id,
        "user_id": user_id,
        "session_id": session_id,
        "branch": branch,
        "local_base_dir": CLONE_DIR,
        "aanalysis_base_dir": ANALYSIS_OUT_DIR,
    }
    logger.debug(f"Starting workflow with parameters: {params}")

    # Validate that graph is provided (should be from FastAPI app.state)
    if graph is None:
        raise ValueError(
            "Workflow graph not provided. "
            "Graph must be initialized at FastAPI startup via lifespan context manager."
        )

    # acquire app_db session
    app_db = next(get_app_db())

    # Use pre-built graph from FastAPI app.state (no need to build here)
    update_rightsize_workflow_graph = graph
    logger.debug("Using pre-built workflow graph from FastAPI app.state")

    # Initialize the state object separately
    try:
        initial_state = RightSizeState(**params)
        await save_workflow_state(
            user_id=user_id, session_id=session_id, state=initial_state, app_db=app_db
        )
        logger.debug("Workflow state initialized and saved to DB.")
    except Exception as e:
        logger.error(f"Failed to initialize state: {e}\n{traceback.format_exc()}")
        raise

    # Invoke the graph with the state at runtime
    try:
        logger.debug(f"About to invoke workflow graph with: {initial_state = }")
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        logger.debug(f"Invoking workflow with checkpoint config: {config}")
        result_state = await update_rightsize_workflow_graph.ainvoke(
            initial_state, config=config
        )

        # Convert dict result to RightSizeState object
        # LangGraph ainvoke returns dict, not Pydantic model
        result_state_obj = RightSizeState(**result_state)

        # Save updated state to WorkflowStateTable (includes HITL pause state)
        # This allows HITL endpoints to find the session and check hitl_pending flag
        await save_workflow_state(
            user_id=user_id,
            session_id=session_id,
            state=result_state_obj,
            app_db=app_db,
        )
        logger.debug(
            f"Workflow state saved after execution (hitl_pending={result_state_obj.hitl_pending})"
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}\n{traceback.format_exc()}")
        raise

    return result_state_obj


# ── Conversational workflow helpers ──────────────────────────────────────────

_GO_APP_PHRASES = ["go app", "start app", "launch app", "begin app", "run app", "go astra", "start astra"]

def _is_go_app_trigger(user_input: str) -> bool:
    """Return True if the user wants to kick off the rightsizing workflow."""
    lowered = user_input.strip().lower()
    return any(phrase in lowered for phrase in _GO_APP_PHRASES)


def _is_collecting_workflow_params(messages: list) -> bool:
    """
    Return True if a previous assistant message asked the user for
    repo_url / app_id / environment — meaning we're mid-param-collection.
    """
    collection_markers = [
        "what is the repository url",
        "please provide the repository url",
        "what is the app id",
        "please provide the app id",
        "which environment",
        "what environment",
        "please specify the environment",
    ]
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(marker in content for marker in collection_markers):
                return True
            break  # Only look at the most recent assistant message
    return False


def _find_current_workflow_start(messages: list) -> int:
    """
    Return the index of the first message that belongs to the *current* workflow
    session — i.e. the assistant welcome message emitted right after the most
    recent 'go app' trigger.

    This prevents params from a previous, completed workflow from bleeding into
    a fresh one when the user says "go app" again.
    """
    # Walk backwards to find the last assistant message that contains the
    # workflow-start greeting marker.
    start_marker = "welcome to **astra rightsizing workflow**"
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            if start_marker in msg.get("content", "").lower():
                return i  # Scan only from this message onwards
    return 0  # No boundary found — scan from the beginning


def _extract_workflow_params_from_history(messages: list) -> dict:
    """
    Walk the conversation history to collect repo_url, app_id, and environment
    that the user has already provided in earlier turns of the *current* workflow.

    Only looks at messages from the most recent workflow-start boundary so that
    params from a previously completed workflow are never reused.
    """
    params = {}

    ask_repo = ["repository url", "repo url"]
    ask_appid = ["app id", "appid", "application id"]
    ask_env = ["which environment", "what environment", "target environment", "specify the environment"]

    # Scope to current workflow only — ignore anything before the last "go app"
    start_idx = _find_current_workflow_start(messages)
    scoped_messages = messages[start_idx:]

    for i, msg in enumerate(scoped_messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").lower()

        # Find the next user reply within the scoped window
        next_user_msg = None
        for j in range(i + 1, len(scoped_messages)):
            if isinstance(scoped_messages[j], dict) and scoped_messages[j].get("role") == "user":
                next_user_msg = scoped_messages[j].get("content", "").strip()
                break

        if next_user_msg is None:
            continue

        if any(k in content for k in ask_repo) and "repo_url" not in params:
            params["repo_url"] = next_user_msg
        elif any(k in content for k in ask_appid) and "app_id" not in params:
            params["app_id"] = next_user_msg
        elif any(k in content for k in ask_env) and "environment" not in params:
            params["environment"] = next_user_msg

    return params


# ── Input validators ─────────────────────────────────────────────────────────

def _validate_repo_url(value: str) -> Optional[str]:
    """
    Return None if valid, or an error message string if invalid.
    A valid repo URL must start with http:// or https://.
    """
    stripped = value.strip()
    if not (stripped.startswith("http://") or stripped.startswith("https://")):
        return (
            "⚠️ That doesn't look like a valid repository URL.\n\n"
            "Please provide a URL that starts with `http://` or `https://` "
            "(e.g. `https://github.com/org/repo.git`)."
        )
    return None


def _validate_app_id(value: str) -> Optional[str]:
    """
    Return None if valid, or an error message string if invalid.
    A valid App ID must start with 'APP' or 'APSVC' (case-insensitive).
    """
    stripped = value.strip().upper()
    if not (stripped.startswith("APP") or stripped.startswith("APSVC")):
        return (
            "⚠️ That doesn't look like a valid App ID.\n\n"
            "App IDs must start with `APP` or `APSVC` "
            "(e.g. `APP0002202` or `APSVC0012`). Please try again."
        )
    return None


def route_node(state: OrchestratorState):
    """Node to route the workflow based on the user input."""
    return state.route["workflow"]


async def general_question_node(state: OrchestratorState):
    """Node to answer general questions."""
    answer = state.route["answer"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def single_tool_node(state: OrchestratorState):
    """Node to call a single tool."""
    tool = state.route["tool"]
    params = state.route["params"]
    logger.debug(f"Inside single_tool_node...Input....: {tool = }, {params = }")
    result = await call_mcp_tool(MCP_SERVER_URL, tool, params)

    new_mcp_logs = state.mcp_logs + [
        {"tool": tool, "params": params, "status": "Success"}
    ]
    new_messages = state.messages + [
        {
            "role": "assistant",
            "content": (
                f"Tool `{tool}` executed with params {params}. Result:\n{result}"
            ),
        },
    ]
    # , {"role": "user", "content": result}]

    logger.debug(f"Inside single_tool_node...Result...: {result}")

    return {"final_answer": result, "mcp_logs": new_mcp_logs, "messages": new_messages}


async def workflow_node(state: OrchestratorState):
    """Node to orchestrate main Astra workflow.

    Conversationally collects repo_url, app_id, and environment one at a time
    before running the workflow.
    """
    if state.route.get("workflow") != "update_rightsize_repo":
        return {"final_answer": "Unsupported workflow"}
    logger.debug(f"Inside workflow_node. OrchestratorState: {state = }")

    # ── Step 1: Gather params from LLM route AND conversation history ─────────
    route_params = state.route.get("params", {}) or {}

    # Merge params already extracted by the LLM router with anything we can
    # recover from the conversation history (for multi-turn collection).
    history_params = _extract_workflow_params_from_history(state.messages)

    # history_params acts as base; route_params overrides (LLM may have parsed
    # the very latest user message, history_params covers earlier turns).
    merged = {**history_params, **route_params}

    repo_url = merged.get("repo_url") or merged.get("repo")
    app_id = merged.get("app_id") or merged.get("appid")
    environment = merged.get("environment") or merged.get("env")

    # ── Step 2: Ask for any missing param (one at a time) ────────────────────
    def _prompt(msg: str):
        new_messages = state.messages + [{"role": "assistant", "content": msg}]
        return {"final_answer": msg, "messages": new_messages}

    if not repo_url:
        logger.info("workflow_node: repo_url missing — prompting user")
        return _prompt(
            "Sure! To kick off the rightsizing workflow I need a few details.\n\n"
            "What is the **Repository URL**? (e.g. `https://github.com/org/repo.git`)"
        )

    # Validate repo_url before moving on
    repo_err = _validate_repo_url(repo_url)
    if repo_err:
        logger.info(f"workflow_node: invalid repo_url '{repo_url}' — re-prompting")
        # Clear the bad value so history extractor won't reuse it next turn.
        # We do this by returning the error as a fresh "ask repo" prompt so
        # _extract_workflow_params_from_history will overwrite it on the next turn.
        return _prompt(repo_err + "\n\nWhat is the **Repository URL**?")

    if not app_id:
        logger.info("workflow_node: app_id missing — prompting user")
        return _prompt(
            f"Got it — repo: `{repo_url}` ✅\n\n"
            "What is the **App ID**? (e.g. `APP0002202`)"
        )

    # Validate app_id before moving on
    appid_err = _validate_app_id(app_id)
    if appid_err:
        logger.info(f"workflow_node: invalid app_id '{app_id}' — re-prompting")
        return _prompt(appid_err + "\n\nWhat is the **App ID**?")

    if not environment:
        logger.info("workflow_node: environment missing — prompting user")
        return _prompt(
            f"Almost there!\n\n"
            f"- Repo: `{repo_url}` ✅\n"
            f"- App ID: `{app_id}` ✅\n\n"
            "Which **environment** would you like to target? (e.g. `dev`, `qa`, `prod`)"
        )

    # ── Step 3: All params collected — run the workflow ──────────────────────
    logger.info(
        f"workflow_node: all params ready — "
        f"repo_url={repo_url}, app_id={app_id}, env={environment}"
    )
    env = environment
    session_id = state.session_id or "s001"
    user_id = state.user_id
    branch = state.branch or "main"

    # Get pre-built workflow graph from state (passed down from FastAPI app)
    graph = state.rightsize_graph
    logger.debug(f"workflow_node: Retrieved graph from state: {graph}")

    if graph is None:
        raise ValueError(
            "Workflow graph not initialized. "
            "Check FastAPI lifespan startup logs for checkpointer initialization errors."
        )

    result = await run_update_rightsize_workflow(
        repo_url=repo_url,
        env=env,
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        branch=branch,
        graph=graph,  # Pass pre-built graph
    )

    # Create user-friendly message instead of dumping full state
    # Extract progress messages from workflow execution
    progress_messages = []
    if result.messages:
        # Filter for assistant messages (progress updates from workflow nodes)
        for msg in result.messages:
            if isinstance(msg, dict):
                if msg.get("type") == "ai" or msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and not content.startswith("⏸"):  # Skip pause messages
                        progress_messages.append(content)

    if result.hitl_checkpoint:
        # Workflow paused at HITL checkpoint
        checkpoint_name = result.hitl_checkpoint
        friendly_message = f"⏸ **Workflow Paused - Approval Required**\n\n"

        # Add progress summary
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Checkpoint:** {checkpoint_name}\n\n"

        if checkpoint_name == "finops_validation":
            rec_count = len(result.recommendations) if result.recommendations else 0
            friendly_message += (
                f"📊 **Status:** Fetched {rec_count} rightsizing recommendation(s) from FinOps\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "analysis_approval":
            var_count = (
                result.analysis_output.get("sizing_variables_count", 0)
                if result.analysis_output
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Analyzed Terraform infrastructure - found {var_count} sizing variable(s)\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "commit_approval":
            files_modified = (
                len(result.hitl_data.get("files_modified", []))
                if result.hitl_data
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Infrastructure updates ready for commit\n"
                f"📄 Modified {files_modified} file(s)\n"
                f"🌿 Branch: {result.feature_branch}\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )

        final_answer = friendly_message
    elif result.error:
        # Workflow failed
        friendly_message = "❌ **Workflow Failed**\n\n"

        # Show progress before failure
        if progress_messages:
            friendly_message += "**Progress before failure:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Error:** {result.error}"
        final_answer = friendly_message
    elif result.pr:
        # Workflow completed successfully
        friendly_message = "✅ **Workflow Complete!**\n\n"

        # Show all progress steps
        if progress_messages:
            friendly_message += "**Completed steps:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**Pull Request:** {result.pr}\n"
            f"**Branch:** {result.feature_branch}\n\n"
            f"👉 **Next Step:** Review and merge the pull request."
        )
        final_answer = friendly_message

    else:
        # Workflow still running or completed without PR
        friendly_message = f"✅ **Rightsizing Workflow Executed**\n\n"

        # Show progress
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**App ID:** {app_id}\n"
            f"**Environment:** {env}\n"
            f"**Session ID:** {session_id}\n\n"
            f"👉 Check workflow status in the sidebar."
        )
        final_answer = friendly_message

    new_mcp_logs = state.mcp_logs + [
        {
            "tool": "update_rightsize_workflow",
            "params": {
                "repo_url": repo_url,
                "environment": env,
                "app_id": app_id,
                "session_id": session_id,
            },
            "status": "Success",
        }
    ]
    new_messages = state.messages + [{"role": "assistant", "content": final_answer}]

    return {
        "final_answer": final_answer,
        "mcp_logs": new_mcp_logs,
        "messages": new_messages,
    }


async def none_node(state: OrchestratorState):
    """Node to handle empty input."""
    answer = state.route["message"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def greeting_node(state: OrchestratorState):
    """Node to handle greeting input.

    If the user said 'go app' (or similar), emit a warm welcome and immediately
    start param collection for the rightsizing workflow.
    """
    if _is_go_app_trigger(state.user_input):
        greeting = (
            "👋 Hey there! Welcome to **Astra Rightsizing Workflow** — let's get started!\n\n"
            "I'll walk you through the process step by step.\n\n"
            "What is the **Repository URL** for the app you'd like to rightsize? "
            "(e.g. `https://github.com/org/repo.git`)"
        )
        new_messages = state.messages + [{"role": "assistant", "content": greeting}]
        return {"final_answer": greeting, "messages": new_messages}

    answer = state.route.get("message", "Hello! How can I help you?")
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def orchestrator(
    user_input: str,
    mcp_logs: list,
    messages: list,
    session_id: str,
    user_id: str,
    rightsize_graph=None,  # Pre-built graph passed from FastAPI app.state
):
    """
    Main orchestrator of the Astra MCP Agent Server.
    Organizes:
        - different types of workflows using LangGraph router pattern
        - external communication using /chat and /tools endpoints.
    """
    # Ensure session_id is not None; should fail later if not provided
    session_id = session_id or str(uuid4())

    # Ensure user_id is not None; should fail later if not provided
    user_id = user_id or "fake_user"

    # Add user's current input to messages BEFORE processing
    # This ensures the user's message is included in the conversation history
    messages_with_input = messages + [{"role": "user", "content": user_input}]

    # ── "go app" shortcut: skip LLM router, go straight to greetings node
    # which will emit a welcome + first param prompt ──────────────────────────
    logger.debug(
        f"In the orchestrator, calling LLM with: {user_input = }, {messages = }"
    )
    if _is_go_app_trigger(user_input):
        logger.info("'go app' trigger detected — routing to greetings node for workflow start")
        route_decision = {
            "workflow": "greetings",
            "message": "",   # greeting_node will build its own message
            "params": {},
        }
    # ── Mid-collection: if last assistant message was a param question,
    # route to workflow_node so it can parse the answer ──────────────────────
    elif _is_collecting_workflow_params(messages):
        logger.info("Mid-param-collection detected — routing to update_rightsize_repo")
        # Use whatever the LLM extracted from this latest message; workflow_node
        # will also scan history for earlier answers.
        route_decision = await llm_client.ask_llm(user_input, messages)
        route_decision["workflow"] = "update_rightsize_repo"
        if "params" not in route_decision or route_decision["params"] is None:
            route_decision["params"] = {}
    else:
        # Normal LLM router call
        route_decision = await llm_client.ask_llm(user_input, messages)
    logger.debug(f"Route: {route_decision = }")

    # Create graph state with accumulated messages (including current user input)
    orch_state = OrchestratorState(
        user_input=user_input,
        session_id=session_id,
        messages=messages_with_input,  # Include user's current message
        mcp_logs=mcp_logs,
        route=route_decision,
        final_answer=None,
        user_id=user_id,
        rightsize_graph=rightsize_graph,  # Pass pre-built graph through state
    )
    logger.debug(f"Initializing with OrchestratorState: {orch_state = }")

    # Build LangGraph
    orchestrator_graph = StateGraph(OrchestratorState)
    orchestrator_graph.add_node("router", lambda s: {"route": s.route})
    orchestrator_graph.set_entry_point("router")

    orchestrator_graph.add_edge("router", orch_state.route["workflow"])

    orchestrator_graph.add_node("none", none_node)
    orchestrator_graph.add_node("greetings", greeting_node)
    orchestrator_graph.add_node("general_question", general_question_node)
    orchestrator_graph.add_node("single_tool", single_tool_node)
    orchestrator_graph.add_node("update_rightsize_repo", workflow_node)

    orchestrator_graph.add_edge("none", END)
    orchestrator_graph.add_edge("greetings", END)
    orchestrator_graph.add_edge("general_question", END)
    orchestrator_graph.add_edge("single_tool", END)
    orchestrator_graph.add_edge("update_rightsize_repo", END)

    logger.debug(f"Orchestrator Graph: {orchestrator_graph}")

    # Compile and execute with no checkpointer
    app = orchestrator_graph.compile()
    logger.info("Orchestrator compiled.")
    logger.debug(f"App: {app} will be invoked with {orch_state = }")
    try:
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id}}
        logger.debug(f"Invoking orchestrator with checkpoint config: {config}")
        result_state = await app.ainvoke(orch_state, config=config)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e} {traceback.format_exc()}")
        result_state = {"final_answer": f"Orchestrator failed: {e}"}
    logger.debug(f"In orchestrator result State before return: {result_state}")
    logger.info("Orchestration complete.")

    return (
        result_state["final_answer"],
        result_state["messages"],
        result_state["mcp_logs"],
    )











"""
Main orchestrator.
Uses langchain, langgraph and LLMto orchestrate the whole workflow in one process.

Currently under construction
Old version of code that must be replaced
"""

# import traceback
from pathlib import Path
import traceback
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Any, Dict, List, Union, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from astra_azure_openai_client import AsyncAzureOpenAIClient
from workflows.rightsize_graph import build_rightsize_graph, RightSizeState
from utils import (
    serialize_result,
    safe_parse_llm_output,
    extract_emails_from_cmdb_metadata,
    extract_finops_data_from_toolresult,
    notify_stakeholders,
)
from config import APP_CFG
from chatbot_logger import logger
from astra_mcp_utils import call_mcp_tool, McpFastAPIClient, list_mcp_tools
from db_utils import save_workflow_state, load_workflow_state, del_workflow_state
from database import get_app_db

# Set CONSTANTS and variables
MCP_TOOLS_URL = APP_CFG.MCP_TOOLS_URL
MCP_SERVER_URL = APP_CFG.MCP_SERVER_URL
MCP_AGENT_URL = APP_CFG.MCP_AGENT_URL
MCP_AGENT_PORT = APP_CFG.MCP_AGENT_PORT
CLONE_DIR = Path(APP_CFG.CLONE_DIR)
ANALYSIS_OUT_DIR = Path(APP_CFG.ANALYSIS_OUT_DIR)

# Async Azure OpenAI client  # TODO: improve config so only AZ part is used here
llm_client = AsyncAzureOpenAIClient(az_cfg=APP_CFG)


# Langgraph setup - state definition, implement nodes
class OrchestratorState(BaseModel):
    """Orchestration state"""

    user_input: str
    session_id: Optional[str] = None
    route: dict = None
    messages: list = []
    mcp_logs: list = []
    final_answer: Any | None = None
    user_id: Optional[str] = None  # TODO: make this mandatory later
    branch: Optional[str] = None
    rightsize_graph: Any = None  # Pre-built workflow graph


# Implement workflow - update_rightsize_workflow
async def run_update_rightsize_workflow(
    # repo_url: str, env: str, app_id: str="APP0002202", branch: str="main"
    repo_url: str,
    env: str,
    app_id: str,
    user_id: str,
    session_id: str = None,
    branch: str = "main",
    graph=None,  # Pre-built graph from FastAPI app.state
) -> RightSizeState:
    """
    Run the Astra MCP Agent workflow for updating rightsize.

    Args:
        repo_url (str): The repository URL.
        env (str): The environment name.
        app_id (str): The application ID.
        session_id (str): The session ID for concurrent workflow execution.
        branch (str): The git branch name.

    Returns:
        RightSizeState: The final state after workflow execution.
    May be create and add to state feature branch name already now
    """
    params = {
        "repo_url": repo_url,
        "environment": env,
        "app_id": app_id,
        "user_id": user_id,
        "session_id": session_id,
        "branch": branch,
        "local_base_dir": CLONE_DIR,
        "aanalysis_base_dir": ANALYSIS_OUT_DIR,
    }
    logger.debug(f"Starting workflow with parameters: {params}")

    # Validate that graph is provided (should be from FastAPI app.state)
    if graph is None:
        raise ValueError(
            "Workflow graph not provided. "
            "Graph must be initialized at FastAPI startup via lifespan context manager."
        )

    # acquire app_db session
    app_db = next(get_app_db())

    # Use pre-built graph from FastAPI app.state (no need to build here)
    update_rightsize_workflow_graph = graph
    logger.debug("Using pre-built workflow graph from FastAPI app.state")

    # Initialize the state object separately
    try:
        initial_state = RightSizeState(**params)
        await save_workflow_state(
            user_id=user_id, session_id=session_id, state=initial_state, app_db=app_db
        )
        logger.debug("Workflow state initialized and saved to DB.")
    except Exception as e:
        logger.error(f"Failed to initialize state: {e}\n{traceback.format_exc()}")
        raise

    # Invoke the graph with the state at runtime
    try:
        logger.debug(f"About to invoke workflow graph with: {initial_state = }")
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        logger.debug(f"Invoking workflow with checkpoint config: {config}")
        result_state = await update_rightsize_workflow_graph.ainvoke(
            initial_state, config=config
        )

        # Convert dict result to RightSizeState object
        # LangGraph ainvoke returns dict, not Pydantic model
        result_state_obj = RightSizeState(**result_state)

        # Save updated state to WorkflowStateTable (includes HITL pause state)
        # This allows HITL endpoints to find the session and check hitl_pending flag
        await save_workflow_state(
            user_id=user_id,
            session_id=session_id,
            state=result_state_obj,
            app_db=app_db,
        )
        logger.debug(
            f"Workflow state saved after execution (hitl_pending={result_state_obj.hitl_pending})"
        )
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}\n{traceback.format_exc()}")
        raise

    return result_state_obj


# ── Conversational workflow helpers ──────────────────────────────────────────

_GO_APP_PHRASES = ["go app", "start app", "launch app", "begin app", "run app", "go astra", "start astra"]

def _is_go_app_trigger(user_input: str) -> bool:
    """Return True if the user wants to kick off the rightsizing workflow."""
    lowered = user_input.strip().lower()
    return any(phrase in lowered for phrase in _GO_APP_PHRASES)


import re

_RIGHTSIZE_APP_PATTERN = re.compile(
    r"right[\-\s]?size\s+(?:app(?:_?id)?|appid)\s+([A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

def _is_rightsize_app_trigger(user_input: str) -> bool:
    """Return True if the user specified an app_id inline with a rightsize command.

    Matches patterns like:
        right-size app 123
        rightsize appid APP0002202
        right size app_id APSVC0012
    """
    return bool(_RIGHTSIZE_APP_PATTERN.search(user_input.strip()))


def _extract_app_id_from_trigger(user_input: str) -> Optional[str]:
    """Extract the app_id from a rightsize-app trigger phrase, or return None."""
    match = _RIGHTSIZE_APP_PATTERN.search(user_input.strip())
    return match.group(1) if match else None


def _is_collecting_workflow_params(messages: list) -> bool:
    """
    Return True if a previous assistant message asked the user for
    repo_url / app_id / environment — meaning we're mid-param-collection.
    """
    collection_markers = [
        "what is the repository url",
        "please provide the repository url",
        "what is the app id",
        "please provide the app id",
        "which environment",
        "what environment",
        "please specify the environment",
    ]
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(marker in content for marker in collection_markers):
                return True
            break  # Only look at the most recent assistant message
    return False


def _find_current_workflow_start(messages: list) -> int:
    """
    Return the index of the first message that belongs to the *current* workflow
    session — i.e. the assistant welcome message emitted right after the most
    recent 'go app' trigger.

    This prevents params from a previous, completed workflow from bleeding into
    a fresh one when the user says "go app" again.
    """
    # Walk backwards to find the last assistant message that contains the
    # workflow-start greeting marker.
    start_markers = [
        "welcome to **astra rightsizing workflow**",
        "let's rightsize app",  # marker from _is_rightsize_app_trigger greeting
    ]
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(marker in content for marker in start_markers):
                return i  # Scan only from this message onwards
    return 0  # No boundary found — scan from the beginning


def _extract_workflow_params_from_history(messages: list) -> dict:
    """
    Walk the conversation history to collect repo_url, app_id, and environment
    that the user has already provided in earlier turns of the *current* workflow.

    Only looks at messages from the most recent workflow-start boundary so that
    params from a previously completed workflow are never reused.
    """
    params = {}

    ask_repo = ["repository url", "repo url"]
    ask_appid = ["app id", "appid", "application id"]
    ask_env = ["which environment", "what environment", "target environment", "specify the environment"]

    # Scope to current workflow only — ignore anything before the last "go app"
    start_idx = _find_current_workflow_start(messages)
    scoped_messages = messages[start_idx:]

    # Pre-seed app_id if the workflow was started with a "rightsize app <id>" trigger
    for msg in scoped_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if _is_rightsize_app_trigger(content):
                extracted = _extract_app_id_from_trigger(content)
                if extracted:
                    params["app_id"] = extracted
                break

    for i, msg in enumerate(scoped_messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").lower()

        # Find the next user reply within the scoped window
        next_user_msg = None
        for j in range(i + 1, len(scoped_messages)):
            if isinstance(scoped_messages[j], dict) and scoped_messages[j].get("role") == "user":
                next_user_msg = scoped_messages[j].get("content", "").strip()
                break

        if next_user_msg is None:
            continue

        if any(k in content for k in ask_repo) and "repo_url" not in params:
            params["repo_url"] = next_user_msg
        elif any(k in content for k in ask_appid) and "app_id" not in params:
            params["app_id"] = next_user_msg
        elif any(k in content for k in ask_env) and "environment" not in params:
            params["environment"] = next_user_msg

    return params


# ── Input validators ─────────────────────────────────────────────────────────

def _validate_repo_url(value: str) -> Optional[str]:
    """
    Return None if valid, or an error message string if invalid.
    A valid repo URL must start with http:// or https://.
    """
    stripped = value.strip()
    if not (stripped.startswith("http://") or stripped.startswith("https://")):
        return (
            "⚠️ That doesn't look like a valid repository URL.\n\n"
            "Please provide a URL that starts with `http://` or `https://` "
            "(e.g. `https://github.com/org/repo.git`)."
        )
    return None


def _validate_app_id(value: str) -> Optional[str]:
    """
    Return None if valid, or an error message string if invalid.
    A valid App ID must start with 'APP' or 'APSVC' (case-insensitive).
    """
    stripped = value.strip().upper()
    if not (stripped.startswith("APP") or stripped.startswith("APSVC")):
        return (
            "⚠️ That doesn't look like a valid App ID.\n\n"
            "App IDs must start with `APP` or `APSVC` "
            "(e.g. `APP0002202` or `APSVC0012`). Please try again."
        )
    return None


def route_node(state: OrchestratorState):
    """Node to route the workflow based on the user input."""
    return state.route["workflow"]


async def general_question_node(state: OrchestratorState):
    """Node to answer general questions."""
    answer = state.route["answer"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def single_tool_node(state: OrchestratorState):
    """Node to call a single tool."""
    tool = state.route["tool"]
    params = state.route["params"]
    logger.debug(f"Inside single_tool_node...Input....: {tool = }, {params = }")
    result = await call_mcp_tool(MCP_SERVER_URL, tool, params)

    new_mcp_logs = state.mcp_logs + [
        {"tool": tool, "params": params, "status": "Success"}
    ]
    new_messages = state.messages + [
        {
            "role": "assistant",
            "content": (
                f"Tool `{tool}` executed with params {params}. Result:\n{result}"
            ),
        },
    ]
    # , {"role": "user", "content": result}]

    logger.debug(f"Inside single_tool_node...Result...: {result}")

    return {"final_answer": result, "mcp_logs": new_mcp_logs, "messages": new_messages}


async def workflow_node(state: OrchestratorState):
    """Node to orchestrate main Astra workflow.

    Conversationally collects repo_url, app_id, and environment one at a time
    before running the workflow.
    """
    if state.route.get("workflow") != "update_rightsize_repo":
        return {"final_answer": "Unsupported workflow"}
    logger.debug(f"Inside workflow_node. OrchestratorState: {state = }")

    # ── Step 1: Gather params from LLM route AND conversation history ─────────
    route_params = state.route.get("params", {}) or {}

    # Merge params already extracted by the LLM router with anything we can
    # recover from the conversation history (for multi-turn collection).
    history_params = _extract_workflow_params_from_history(state.messages)

    # history_params acts as base; route_params overrides (LLM may have parsed
    # the very latest user message, history_params covers earlier turns).
    merged = {**history_params, **route_params}

    repo_url = merged.get("repo_url") or merged.get("repo")
    app_id = merged.get("app_id") or merged.get("appid")
    environment = merged.get("environment") or merged.get("env")

    # ── Step 2: Ask for any missing param (one at a time) ────────────────────
    def _prompt(msg: str):
        new_messages = state.messages + [{"role": "assistant", "content": msg}]
        return {"final_answer": msg, "messages": new_messages}

    if not repo_url:
        logger.info("workflow_node: repo_url missing — prompting user")
        return _prompt(
            "Sure! To kick off the rightsizing workflow I need a few details.\n\n"
            "What is the **Repository URL**? (e.g. `https://github.com/org/repo.git`)"
        )

    # Validate repo_url before moving on
    repo_err = _validate_repo_url(repo_url)
    if repo_err:
        logger.info(f"workflow_node: invalid repo_url '{repo_url}' — re-prompting")
        # Clear the bad value so history extractor won't reuse it next turn.
        # We do this by returning the error as a fresh "ask repo" prompt so
        # _extract_workflow_params_from_history will overwrite it on the next turn.
        return _prompt(repo_err + "\n\nWhat is the **Repository URL**?")

    if not app_id:
        logger.info("workflow_node: app_id missing — prompting user")
        return _prompt(
            f"Got it — repo: `{repo_url}` ✅\n\n"
            "What is the **App ID**? (e.g. `APP0002202`)"
        )

    # Validate app_id before moving on
    appid_err = _validate_app_id(app_id)
    if appid_err:
        logger.info(f"workflow_node: invalid app_id '{app_id}' — re-prompting")
        return _prompt(appid_err + "\n\nWhat is the **App ID**?")

    if not environment:
        logger.info("workflow_node: environment missing — prompting user")
        return _prompt(
            f"Almost there!\n\n"
            f"- Repo: `{repo_url}` ✅\n"
            f"- App ID: `{app_id}` ✅\n\n"
            "Which **environment** would you like to target? (e.g. `dev`, `qa`, `prod`)"
        )

    # ── Step 3: All params collected — run the workflow ──────────────────────
    logger.info(
        f"workflow_node: all params ready — "
        f"repo_url={repo_url}, app_id={app_id}, env={environment}"
    )
    env = environment
    session_id = state.session_id or "s001"
    user_id = state.user_id
    branch = state.branch or "main"

    # Get pre-built workflow graph from state (passed down from FastAPI app)
    graph = state.rightsize_graph
    logger.debug(f"workflow_node: Retrieved graph from state: {graph}")

    if graph is None:
        raise ValueError(
            "Workflow graph not initialized. "
            "Check FastAPI lifespan startup logs for checkpointer initialization errors."
        )

    result = await run_update_rightsize_workflow(
        repo_url=repo_url,
        env=env,
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        branch=branch,
        graph=graph,  # Pass pre-built graph
    )

    # Create user-friendly message instead of dumping full state
    # Extract progress messages from workflow execution
    progress_messages = []
    if result.messages:
        # Filter for assistant messages (progress updates from workflow nodes)
        for msg in result.messages:
            if isinstance(msg, dict):
                if msg.get("type") == "ai" or msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and not content.startswith("⏸"):  # Skip pause messages
                        progress_messages.append(content)

    if result.hitl_checkpoint:
        # Workflow paused at HITL checkpoint
        checkpoint_name = result.hitl_checkpoint
        friendly_message = f"⏸ **Workflow Paused - Approval Required**\n\n"

        # Add progress summary
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Checkpoint:** {checkpoint_name}\n\n"

        if checkpoint_name == "finops_validation":
            rec_count = len(result.recommendations) if result.recommendations else 0
            friendly_message += (
                f"📊 **Status:** Fetched {rec_count} rightsizing recommendation(s) from FinOps\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "analysis_approval":
            var_count = (
                result.analysis_output.get("sizing_variables_count", 0)
                if result.analysis_output
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Analyzed Terraform infrastructure - found {var_count} sizing variable(s)\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )
        elif checkpoint_name == "commit_approval":
            files_modified = (
                len(result.hitl_data.get("files_modified", []))
                if result.hitl_data
                else 0
            )
            friendly_message += (
                f"📋 **Status:** Infrastructure updates ready for commit\n"
                f"📄 Modified {files_modified} file(s)\n"
                f"🌿 Branch: {result.feature_branch}\n\n"
                f"👉 **Next Step:** Review and approve/reject using natural language in a chat."
            )

        final_answer = friendly_message
    elif result.error:
        # Workflow failed
        friendly_message = "❌ **Workflow Failed**\n\n"

        # Show progress before failure
        if progress_messages:
            friendly_message += "**Progress before failure:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += f"**Error:** {result.error}"
        final_answer = friendly_message
    elif result.pr:
        # Workflow completed successfully
        friendly_message = "✅ **Workflow Complete!**\n\n"

        # Show all progress steps
        if progress_messages:
            friendly_message += "**Completed steps:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**Pull Request:** {result.pr}\n"
            f"**Branch:** {result.feature_branch}\n\n"
            f"👉 **Next Step:** Review and merge the pull request."
        )
        final_answer = friendly_message

    else:
        # Workflow still running or completed without PR
        friendly_message = f"✅ **Rightsizing Workflow Executed**\n\n"

        # Show progress
        if progress_messages:
            friendly_message += "**Progress:**\n"
            for msg in progress_messages:
                friendly_message += f"✅ {msg}\n"
            friendly_message += "\n"

        friendly_message += (
            f"**App ID:** {app_id}\n"
            f"**Environment:** {env}\n"
            f"**Session ID:** {session_id}\n\n"
            f"👉 Check workflow status in the sidebar."
        )
        final_answer = friendly_message

    new_mcp_logs = state.mcp_logs + [
        {
            "tool": "update_rightsize_workflow",
            "params": {
                "repo_url": repo_url,
                "environment": env,
                "app_id": app_id,
                "session_id": session_id,
            },
            "status": "Success",
        }
    ]
    new_messages = state.messages + [{"role": "assistant", "content": final_answer}]

    return {
        "final_answer": final_answer,
        "mcp_logs": new_mcp_logs,
        "messages": new_messages,
    }


async def none_node(state: OrchestratorState):
    """Node to handle empty input."""
    answer = state.route["message"]
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def greeting_node(state: OrchestratorState):
    """Node to handle greeting input.

    If the user said 'go app' (or similar), emit a warm welcome and immediately
    start param collection for the rightsizing workflow.
    """
    if _is_go_app_trigger(state.user_input):
        greeting = (
            "👋 Hey there! Welcome to **Astra Rightsizing Workflow** — let's get started!\n\n"
            "I'll walk you through the process step by step.\n\n"
            "What is the **Repository URL** for the app you'd like to rightsize? "
            "(e.g. `https://github.com/org/repo.git`)"
        )
        new_messages = state.messages + [{"role": "assistant", "content": greeting}]
        return {"final_answer": greeting, "messages": new_messages}

    if _is_rightsize_app_trigger(state.user_input):
        app_id = _extract_app_id_from_trigger(state.user_input)
        greeting = (
            f"👋 Let's rightsize app **{app_id}**!\n\n"
            "What is the **Repository URL** for this app? "
            "(e.g. `https://github.com/org/repo.git`)"
        )
        new_messages = state.messages + [{"role": "assistant", "content": greeting}]
        return {"final_answer": greeting, "messages": new_messages}

    answer = state.route.get("message", "Hello! How can I help you?")
    new_messages = state.messages + [{"role": "assistant", "content": answer}]
    return {"final_answer": answer, "messages": new_messages}


async def orchestrator(
    user_input: str,
    mcp_logs: list,
    messages: list,
    session_id: str,
    user_id: str,
    rightsize_graph=None,  # Pre-built graph passed from FastAPI app.state
):
    """
    Main orchestrator of the Astra MCP Agent Server.
    Organizes:
        - different types of workflows using LangGraph router pattern
        - external communication using /chat and /tools endpoints.
    """
    # Ensure session_id is not None; should fail later if not provided
    session_id = session_id or str(uuid4())

    # Ensure user_id is not None; should fail later if not provided
    user_id = user_id or "fake_user"

    # Add user's current input to messages BEFORE processing
    # This ensures the user's message is included in the conversation history
    messages_with_input = messages + [{"role": "user", "content": user_input}]

    # ── "go app" shortcut: skip LLM router, go straight to greetings node
    # which will emit a welcome + first param prompt ──────────────────────────
    logger.debug(
        f"In the orchestrator, calling LLM with: {user_input = }, {messages = }"
    )
    if _is_go_app_trigger(user_input):
        logger.info("'go app' trigger detected — routing to greetings node for workflow start")
        route_decision = {
            "workflow": "greetings",
            "message": "",   # greeting_node will build its own message
            "params": {},
        }
    elif _is_rightsize_app_trigger(user_input):
        logger.info("'rightsize app <id>' trigger detected — routing to greetings node")
        app_id = _extract_app_id_from_trigger(user_input)
        route_decision = {
            "workflow": "greetings",
            "message": "",
            "params": {"app_id": app_id},
        }
    # ── Mid-collection: if last assistant message was a param question,
    # route to workflow_node so it can parse the answer ──────────────────────
    elif _is_collecting_workflow_params(messages):
        logger.info("Mid-param-collection detected — routing to update_rightsize_repo")
        # Use whatever the LLM extracted from this latest message; workflow_node
        # will also scan history for earlier answers.
        route_decision = await llm_client.ask_llm(user_input, messages)
        route_decision["workflow"] = "update_rightsize_repo"
        if "params" not in route_decision or route_decision["params"] is None:
            route_decision["params"] = {}
    else:
        # Normal LLM router call
        route_decision = await llm_client.ask_llm(user_input, messages)
    logger.debug(f"Route: {route_decision = }")

    # Create graph state with accumulated messages (including current user input)
    orch_state = OrchestratorState(
        user_input=user_input,
        session_id=session_id,
        messages=messages_with_input,  # Include user's current message
        mcp_logs=mcp_logs,
        route=route_decision,
        final_answer=None,
        user_id=user_id,
        rightsize_graph=rightsize_graph,  # Pass pre-built graph through state
    )
    logger.debug(f"Initializing with OrchestratorState: {orch_state = }")

    # Build LangGraph
    orchestrator_graph = StateGraph(OrchestratorState)
    orchestrator_graph.add_node("router", lambda s: {"route": s.route})
    orchestrator_graph.set_entry_point("router")

    orchestrator_graph.add_edge("router", orch_state.route["workflow"])

    orchestrator_graph.add_node("none", none_node)
    orchestrator_graph.add_node("greetings", greeting_node)
    orchestrator_graph.add_node("general_question", general_question_node)
    orchestrator_graph.add_node("single_tool", single_tool_node)
    orchestrator_graph.add_node("update_rightsize_repo", workflow_node)

    orchestrator_graph.add_edge("none", END)
    orchestrator_graph.add_edge("greetings", END)
    orchestrator_graph.add_edge("general_question", END)
    orchestrator_graph.add_edge("single_tool", END)
    orchestrator_graph.add_edge("update_rightsize_repo", END)

    logger.debug(f"Orchestrator Graph: {orchestrator_graph}")

    # Compile and execute with no checkpointer
    app = orchestrator_graph.compile()
    logger.info("Orchestrator compiled.")
    logger.debug(f"App: {app} will be invoked with {orch_state = }")
    try:
        # Bug fix: Add config with thread_id for LangGraph checkpointing
        config = {"configurable": {"thread_id": session_id}}
        logger.debug(f"Invoking orchestrator with checkpoint config: {config}")
        result_state = await app.ainvoke(orch_state, config=config)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e} {traceback.format_exc()}")
        result_state = {"final_answer": f"Orchestrator failed: {e}"}
    logger.debug(f"In orchestrator result State before return: {result_state}")
    logger.info("Orchestration complete.")

    return (
        result_state["final_answer"],
        result_state["messages"],
        result_state["mcp_logs"],
    )
