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
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def match_finops_to_terraform(tf_file, finops_file, output_file="matched_results.json"):
    tf_data = load_json_file(tf_file)
    finops_data = load_json_file(finops_file)

    if tf_data is None or finops_data is None:
        return

    sizing_vars = tf_data.get("sizing_variables", [])
    matched_results = []
    THRESHOLD = 0.7

    for finops in finops_data:
        best_match = None
        highest_score = -1

        for tf in sizing_vars:
            # 1. Context Match (Environment)
            env_match = 1.0 if tf.get("environment") == finops.get("environment_name") else 0.0
            
            # 2. Semantic Match: id -> variable_name
            name_score = get_similarity(tf.get("id"), finops.get("variable_name"))
            
            # 3. Semantic Match: category -> variable_description
            desc_score = get_similarity(tf.get("category"), finops.get("variable_description"))
            
            # 4. Value Match: value -> variable_value_old
            val_match = 1.0 if str(tf.get("value")) == str(finops.get("variable_value_old")) else 0.0

            # Weighted Score (Env: 40%, Name: 30%, Desc: 20%, Val: 10%)
            total_score = (env_match * 0.4) + (name_score * 0.3) + (desc_score * 0.2) + (val_match * 0.1)

            if total_score > highest_score:
                highest_score = total_score
                best_match = tf

        # (1) Only consider if match score > 0.7
        if best_match and highest_score >= THRESHOLD:
            matched_results.append({
                "resource_group_name": best_match.get("resource_group"),
                "resource_name": best_match.get("service"),
                "variable_name": finops.get("variable_name"),
                "variable_value_old": best_match.get("value"), # Value from Terraform
                "variable_value_new": finops.get("variable_value_new"), # Recommendation from FinOps
                "confidence": round(highest_score, 4)
            })

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(matched_results, f, indent=2)

    # (2) Print results with specific columns
    print(f"{'RG Name':<30} | {'Resource':<20} | {'Old Val':<8} | {'New Val':<8} | {'Score'}")
    print("-" * 85)
    for res in matched_results:
        print(f"{str(res['resource_group_name']):<30} | "
              f"{str(res['resource_name']):<20} | "
              f"{str(res['variable_value_old']):<8} | "
              f"{str(res['variable_value_new']):<8} | "
              f"{res['confidence']}")

if __name__ == "__main__":
    match_finops_to_terraform("terraform_analysis.json", "finops_data.json")
