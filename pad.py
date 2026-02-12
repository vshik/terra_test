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
        with open(file_path, 'r', encoding='utf-8') as f:
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
            
            # 2. Semantic Match: id (TF) -> variable_name (FinOps)
            name_score = get_similarity(tf.get("id"), finops.get("variable_name"))
            
            # 3. Semantic Match: category (TF) -> variable_description (FinOps)
            desc_score = get_similarity(tf.get("category"), finops.get("variable_description"))
            
            # 4. Value Match: value (TF) -> variable_value_old (FinOps)
            val_match = 1.0 if str(tf.get("value")) == str(finops.get("variable_value_old")) else 0.0

            # Weighting Logic
            total_score = (env_match * 0.4) + (name_score * 0.3) + (desc_score * 0.2) + (val_match * 0.1)

            if total_score > highest_score:
                highest_score = total_score
                best_match = tf

        # Only include if match score strictly > 0.7
        if best_match and highest_score > THRESHOLD:
            matched_results.append({
                "finops_resource_group": finops.get("resource_group_name"),
                "finops_resource_name": finops.get("resource_name"),
                "finops_variable": finops.get("variable_name"),
                "terraform_id": best_match.get("id"),
                "variable_value_old": best_match.get("value"),
                "variable_value_new": finops.get("variable_value_new"),
                "confidence": round(highest_score, 4)
            })

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_results, f, indent=2)

    # Output with FinOps-sourced resource details
    header = f"{'RG (FinOps)':<35} | {'Resource (FinOps)':<30} | {'Var Name':<12} | {'Old':<6} | {'New':<6} | {'Score'}"
    print(header)
    print("-" * len(header))
    
    for res in matched_results:
        print(f"{str(res['finops_resource_group']):<35} | "
              f"{str(res['finops_resource_name']):<30} | "
              f"{str(res['finops_variable']):<12} | "
              f"{str(res['variable_value_old']):<6} | "
              f"{str(res['variable_value_new']):<6} | "
              f"{res['confidence']}")

if __name__ == "__main__":
    match_finops_to_terraform("terraform_analysis.json", "finops_data.json")

if __name__ == "__main__":
    match_finops_to_terraform("terraform_analysis.json", "finops_data.json")




# Turbonomic to Terraform Mapping Guide for Azure Resources

## Overview
This document maps Turbonomic's standardized recommendation fields to Terraform resource attributes and common variable naming patterns for Azure resources.

---

## Virtual Machines (VMs)

### Resource Types
- `azurerm_virtual_machine`
- `azurerm_linux_virtual_machine`
- `azurerm_windows_virtual_machine`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended VM size | `vm_size` / `size` | `vm_size`, `usr_vm_size`, `instance_type`, `sku` | Standard_D2s_v3, Standard_E4s_v5 |
| `instanceType_before` | Current VM size | `vm_size` / `size` | (current value) | Standard_D4s_v3 |
| `vCPU_after` | Recommended vCPUs | (derived from vm_size) | N/A (informational) | 2, 4, 8 |
| `vMem_after` | Recommended memory (GB) | (derived from vm_size) | N/A (informational) | 8, 16, 32 |
| `numVMs` | Current VM count | `count` / `instances` | `vm_count`, `usr_vm_count`, `instance_count` | 1, 3, 5 |

### Terraform Example
```hcl
resource "azurerm_linux_virtual_machine" "app" {
  name                = "app-vm-${count.index}"
  count               = var.usr_vm_count           # Maps to: numVMs
  size                = var.usr_vm_size            # Maps to: instanceType_after
  resource_group_name = var.resource_group_name
  location            = var.location
  # ... other config
}
```

---

## Virtual Machine Scale Sets (VMSS)

### Resource Type
- `azurerm_virtual_machine_scale_set`
- `azurerm_linux_virtual_machine_scale_set`
- `azurerm_windows_virtual_machine_scale_set`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended VM size | `sku` / `vm_size` | `vmss_sku`, `usr_vmss_size`, `sku_name` | Standard_D2s_v3 |
| `instanceCount_after` | Recommended instance count | `instances` / `capacity` | `vmss_capacity`, `usr_instance_count`, `min_instances`, `max_instances` | 3, 10, 20 |
| `storageAmount_after` | Recommended storage (GB) | `storage_profile_os_disk.disk_size_gb` | `os_disk_size`, `usr_disk_size_gb` | 128, 256, 512 |

### Terraform Example
```hcl
resource "azurerm_linux_virtual_machine_scale_set" "web" {
  name                = "web-vmss"
  sku                 = var.usr_vmss_size         # Maps to: instanceType_after
  instances           = var.usr_instance_count    # Maps to: instanceCount_after
  resource_group_name = var.resource_group_name
  location            = var.location
  # ... other config
}
```

---

## Azure SQL Database

### Resource Types
- `azurerm_mssql_database`
- `azurerm_sql_database`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended SKU/tier | `sku_name` | `db_sku`, `usr_db_sku`, `database_sku`, `sku_name` | S0, S1, P1, GP_Gen5_2 |
| `storageAmount_after` | Recommended storage (GB) | `max_size_gb` | `max_size_gb`, `usr_db_storage`, `storage_gb` | 10, 50, 250, 500 |
| `vCPU_after` | Recommended vCores | (part of sku_name) | N/A (embedded in SKU) | 2, 4, 8 (in GP_Gen5_2) |
| `DTU_after` | Recommended DTUs | (part of sku_name) | N/A (embedded in SKU) | 10, 20, 50 (in S0, S1, S2) |

### SKU Name Formats
- **DTU-based**: `S0`, `S1`, `S2`, `P1`, `P2`, `P4`
- **vCore-based**: `GP_Gen5_2`, `BC_Gen5_4`, `HS_Gen5_8`

### Terraform Example
```hcl
resource "azurerm_mssql_database" "main" {
  name        = "main-db"
  server_id   = azurerm_mssql_server.main.id
  sku_name    = var.usr_db_sku              # Maps to: instanceType_after
  max_size_gb = var.usr_db_storage          # Maps to: storageAmount_after
  # ... other config
}
```

---

## Azure Database for PostgreSQL / MySQL

### Resource Types
- `azurerm_postgresql_server`
- `azurerm_postgresql_flexible_server`
- `azurerm_mysql_server`
- `azurerm_mysql_flexible_server`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended SKU | `sku_name` | `db_sku`, `usr_postgres_sku`, `sku_name` | B_Gen5_1, GP_Gen5_2, MO_Gen5_4 |
| `storageAmount_after` | Recommended storage (MB) | `storage_mb` | `storage_mb`, `usr_db_storage_mb` | 5120, 10240, 102400 |
| `vCPU_after` | Recommended vCores | (part of sku_name) | N/A | 1, 2, 4, 8 |
| `backup_retention_days` | Backup retention | `backup_retention_days` | `backup_retention`, `usr_backup_days` | 7, 14, 35 |

### Terraform Example
```hcl
resource "azurerm_postgresql_flexible_server" "main" {
  name                = "postgres-server"
  sku_name            = var.usr_postgres_sku        # Maps to: instanceType_after
  storage_mb          = var.usr_db_storage_mb       # Maps to: storageAmount_after
  resource_group_name = var.resource_group_name
  location            = var.location
  # ... other config
}
```

---

## Azure Kubernetes Service (AKS)

### Resource Type
- `azurerm_kubernetes_cluster`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended node VM size | `default_node_pool.vm_size` | `node_vm_size`, `usr_aks_node_size`, `agent_vm_size` | Standard_D2s_v3, Standard_D4s_v3 |
| `instanceCount_after` | Recommended node count | `default_node_pool.node_count` / `min_count` / `max_count` | `node_count`, `usr_node_count`, `min_nodes`, `max_nodes` | 3, 5, 10 |
| `storageAmount_after` | OS disk size (GB) | `default_node_pool.os_disk_size_gb` | `os_disk_size_gb`, `usr_disk_size` | 100, 128, 256 |

### Terraform Example
```hcl
resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-cluster"
  resource_group_name = var.resource_group_name
  location            = var.location
  dns_prefix          = "aks"
  
  default_node_pool {
    name                = "default"
    vm_size             = var.usr_aks_node_size    # Maps to: instanceType_after
    node_count          = var.usr_node_count       # Maps to: instanceCount_after
    enable_auto_scaling = true
    min_count           = var.usr_min_nodes
    max_count           = var.usr_max_nodes
    os_disk_size_gb     = var.usr_disk_size        # Maps to: storageAmount_after
  }
  # ... other config
}
```

---

## Azure App Service / Web Apps

### Resource Types
- `azurerm_app_service`
- `azurerm_linux_web_app`
- `azurerm_windows_web_app`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended App Service Plan SKU | `azurerm_app_service_plan.sku.tier` + `sku.size` | `app_service_sku`, `usr_plan_sku`, `sku_tier`, `sku_size` | B1, S1, P1v2, P2v3 |
| `instanceCount_after` | Recommended instance count | `azurerm_app_service_plan.sku.capacity` | `instance_capacity`, `usr_app_instances` | 1, 2, 3 |

### App Service Plan SKU Format
- **Tier + Size**: `Basic` (B1, B2), `Standard` (S1, S2, S3), `Premium` (P1v2, P2v2, P3v2), `PremiumV3` (P1v3, P2v3)

### Terraform Example
```hcl
resource "azurerm_service_plan" "main" {
  name                = "app-service-plan"
  resource_group_name = var.resource_group_name
  location            = var.location
  os_type             = "Linux"
  sku_name            = var.usr_plan_sku           # Maps to: instanceType_after
  worker_count        = var.usr_app_instances      # Maps to: instanceCount_after
}

resource "azurerm_linux_web_app" "main" {
  name                = "web-app"
  service_plan_id     = azurerm_service_plan.main.id
  resource_group_name = var.resource_group_name
  location            = var.location
  # ... other config
}
```

---

## Azure Container Instances (ACI)

### Resource Type
- `azurerm_container_group`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `vCPU_after` | Recommended CPU cores | `container.cpu` | `container_cpu`, `usr_cpu_cores` | 0.5, 1, 2, 4 |
| `vMem_after` | Recommended memory (GB) | `container.memory` | `container_memory`, `usr_memory_gb` | 0.5, 1, 2, 4 |
| `instanceCount_after` | Container replicas | Count of containers in group | `container_count`, `usr_container_count` | 1, 2, 3 |

### Terraform Example
```hcl
resource "azurerm_container_group" "main" {
  name                = "aci-group"
  resource_group_name = var.resource_group_name
  location            = var.location
  os_type             = "Linux"
  
  container {
    name   = "app"
    image  = "nginx:latest"
    cpu    = var.usr_cpu_cores         # Maps to: vCPU_after
    memory = var.usr_memory_gb         # Maps to: vMem_after
  }
  # ... other config
}
```

---

## Azure Storage Accounts

### Resource Type
- `azurerm_storage_account`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended account tier/replication | `account_tier` + `account_replication_type` | `storage_tier`, `usr_storage_tier`, `replication_type` | Standard_LRS, Premium_ZRS |
| `storageAmount_after` | Used/recommended capacity | (informational only) | N/A | 100GB, 1TB |
| `performance_tier` | Performance tier | `account_tier` | `storage_account_tier`, `usr_tier` | Standard, Premium |

### Account Types
- **Tier**: `Standard`, `Premium`
- **Replication**: `LRS`, `GRS`, `RAGRS`, `ZRS`, `GZRS`, `RAGZRS`

### Terraform Example
```hcl
resource "azurerm_storage_account" "main" {
  name                     = "storageaccount"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = var.usr_storage_tier        # Maps to: instanceType_after (tier part)
  account_replication_type = var.usr_replication_type    # Maps to: instanceType_after (replication part)
  # ... other config
}
```

---

## Azure Managed Disks

### Resource Type
- `azurerm_managed_disk`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended disk type | `storage_account_type` | `disk_type`, `usr_disk_sku`, `disk_sku` | Standard_LRS, Premium_SSD, StandardSSD_ZRS |
| `storageAmount_after` | Recommended disk size (GB) | `disk_size_gb` | `disk_size_gb`, `usr_disk_size` | 32, 64, 128, 256, 512, 1024 |
| `diskIOPS_after` | Recommended IOPS | (derived from disk type/size) | N/A | 500, 2300, 5000 |
| `throughput_after` | Recommended throughput (MB/s) | (derived from disk type/size) | N/A | 60, 150, 200 |

### Disk Types
- `Standard_LRS`, `StandardSSD_LRS`, `StandardSSD_ZRS`
- `Premium_LRS`, `Premium_ZRS`
- `PremiumV2_LRS`, `UltraSSD_LRS`

### Terraform Example
```hcl
resource "azurerm_managed_disk" "data" {
  name                 = "data-disk"
  resource_group_name  = var.resource_group_name
  location             = var.location
  storage_account_type = var.usr_disk_sku         # Maps to: instanceType_after
  disk_size_gb         = var.usr_disk_size        # Maps to: storageAmount_after
  create_option        = "Empty"
  # ... other config
}
```

---

## Azure Redis Cache

### Resource Type
- `azurerm_redis_cache`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Recommended SKU (family + capacity) | `family` + `capacity` | `redis_sku`, `usr_redis_capacity`, `sku_name` | C0, C1, P1, P2 |
| `vMem_after` | Cache memory (GB) | (derived from capacity) | N/A | 0.25, 1, 6, 13 |
| `sku_tier` | SKU tier | `sku_name` | `redis_tier`, `usr_cache_tier` | Basic, Standard, Premium |

### Redis SKU Format
- **Basic/Standard**: C0-C6 (250MB - 53GB)
- **Premium**: P1-P5 (6GB - 120GB)

### Terraform Example
```hcl
resource "azurerm_redis_cache" "main" {
  name                = "redis-cache"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku_name            = var.usr_cache_tier        # Maps to: sku_tier (Basic/Standard/Premium)
  family              = var.usr_redis_family      # Maps to: instanceType_after (C or P)
  capacity            = var.usr_redis_capacity    # Maps to: instanceType_after (0-6 or 1-5)
  # ... other config
}
```

---

## Azure Cosmos DB

### Resource Type
- `azurerm_cosmosdb_account`
- `azurerm_cosmosdb_sql_database`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `throughput_after` | Recommended RU/s | `throughput` | `cosmos_throughput`, `usr_ru_per_second`, `throughput` | 400, 1000, 4000, 10000 |
| `storageAmount_after` | Storage size (GB) | (informational/autoscale) | N/A | 10, 100, 1000 |
| `autoscale_max_throughput` | Max autoscale RU/s | `autoscale_settings.max_throughput` | `max_throughput`, `usr_max_rus` | 1000, 4000, 10000 |

### Terraform Example
```hcl
resource "azurerm_cosmosdb_sql_database" "main" {
  name                = "cosmos-db"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  throughput          = var.usr_ru_per_second     # Maps to: throughput_after
  
  # OR for autoscale:
  # autoscale_settings {
  #   max_throughput = var.usr_max_rus            # Maps to: autoscale_max_throughput
  # }
}
```

---

## Azure Load Balancer

### Resource Type
- `azurerm_lb`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | SKU type | `sku` | `lb_sku`, `usr_lb_sku` | Basic, Standard, Gateway |
| `sku_tier` | SKU tier | `sku_tier` | `lb_tier`, `usr_lb_tier` | Regional, Global |

### Terraform Example
```hcl
resource "azurerm_lb" "main" {
  name                = "load-balancer"
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = var.usr_lb_sku           # Maps to: instanceType_after
  sku_tier            = var.usr_lb_tier          # Maps to: sku_tier
  # ... other config
}
```

---

## Azure Application Gateway

### Resource Type
- `azurerm_application_gateway`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | SKU name/tier | `sku.name` + `sku.tier` | `appgw_sku`, `usr_gateway_sku` | Standard_Small, WAF_Medium, Standard_v2 |
| `instanceCount_after` | Instance capacity | `sku.capacity` / `autoscale_configuration` | `gateway_capacity`, `usr_appgw_capacity`, `min_capacity`, `max_capacity` | 2, 3, 10 |

### Application Gateway SKUs
- **V1**: Standard_Small, Standard_Medium, Standard_Large, WAF_Medium, WAF_Large
- **V2**: Standard_v2, WAF_v2

### Terraform Example
```hcl
resource "azurerm_application_gateway" "main" {
  name                = "app-gateway"
  resource_group_name = var.resource_group_name
  location            = var.location
  
  sku {
    name     = var.usr_gateway_sku         # Maps to: instanceType_after
    tier     = var.usr_gateway_tier
    capacity = var.usr_appgw_capacity      # Maps to: instanceCount_after
  }
  
  # OR for autoscaling (v2 only):
  # autoscale_configuration {
  #   min_capacity = var.usr_min_capacity
  #   max_capacity = var.usr_max_capacity
  # }
  # ... other config
}
```

---

## Azure Functions

### Resource Types
- `azurerm_function_app`
- `azurerm_linux_function_app`
- `azurerm_windows_function_app`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | App Service Plan SKU | `service_plan_id` (references plan SKU) | `function_sku`, `usr_function_plan_sku` | Y1, EP1, EP2, P1v2, P2v2 |
| `instanceCount_after` | Pre-warmed instances | `app_settings["WEBSITE_MAX_DYNAMIC_APPLICATION_SCALE_OUT"]` | `max_scale_out`, `usr_max_instances` | 10, 20, 100 |

### Function Plan Types
- **Consumption**: Y1 (dynamic scaling)
- **Premium**: EP1, EP2, EP3 (pre-warmed instances)
- **Dedicated**: Same as App Service (B1, S1, P1v2, etc.)

### Terraform Example
```hcl
resource "azurerm_service_plan" "functions" {
  name                = "functions-plan"
  resource_group_name = var.resource_group_name
  location            = var.location
  os_type             = "Linux"
  sku_name            = var.usr_function_plan_sku    # Maps to: instanceType_after
}

resource "azurerm_linux_function_app" "main" {
  name                = "function-app"
  service_plan_id     = azurerm_service_plan.functions.id
  resource_group_name = var.resource_group_name
  location            = var.location
  
  app_settings = {
    "WEBSITE_MAX_DYNAMIC_APPLICATION_SCALE_OUT" = var.usr_max_instances  # Maps to: instanceCount_after
  }
  # ... other config
}
```

---

## Azure Virtual Network Gateway (VPN Gateway)

### Resource Type
- `azurerm_virtual_network_gateway`

| Turbonomic Field | Description | Terraform Attribute | Common Variable Names | Example Values |
|-----------------|-------------|---------------------|----------------------|----------------|
| `instanceType_after` | Gateway SKU | `sku` | `vpn_sku`, `usr_gateway_sku` | Basic, VpnGw1, VpnGw2, VpnGw3 |
| `vpn_type` | VPN type | `vpn_type` | `vpn_type`, `usr_vpn_type` | RouteBased, PolicyBased |

### VPN Gateway SKUs
- **Basic**: Basic (legacy)
- **Generation 1**: VpnGw1, VpnGw2, VpnGw3
- **Generation 2**: VpnGw2, VpnGw3, VpnGw4, VpnGw5

### Terraform Example
```hcl
resource "azurerm_virtual_network_gateway" "main" {
  name                = "vpn-gateway"
  resource_group_name = var.resource_group_name
  location            = var.location
  type                = "Vpn"
  vpn_type            = var.usr_vpn_type         # Maps to: vpn_type
  sku                 = var.usr_gateway_sku      # Maps to: instanceType_after
  # ... other config
}
```

---

## Summary Table: Common Field Mappings Across Resources

| Turbonomic Field | Typical Terraform Attributes | Purpose |
|-----------------|----------------------------|---------|
| `instanceType_after` | `size`, `sku`, `sku_name`, `vm_size`, `family` + `capacity` | Size/SKU recommendation |
| `instanceType_before` | (current value of above) | Current size/SKU |
| `instanceCount_after` | `count`, `instances`, `capacity`, `node_count`, `worker_count` | Scaling recommendation |
| `vCPU_after` | (derived from size/sku) | CPU cores (informational) |
| `vMem_after` | (derived from size/sku) or `memory` | Memory in GB (informational) |
| `storageAmount_after` | `disk_size_gb`, `max_size_gb`, `storage_mb`, `os_disk_size_gb` | Storage size recommendation |
| `throughput_after` | `throughput` (Cosmos DB), derived from disk type | RU/s or MB/s recommendation |
| `diskIOPS_after` | (derived from disk type/size) | IOPS recommendation |

---

## Best Practices for Mapping

### 1. **Use Terraform State as Source of Truth**
```bash
# List all resources
terraform state list

# Show specific resource details
terraform state show azurerm_linux_virtual_machine.app

# Export state to JSON for parsing
terraform show -json > terraform-state.json
```

### 2. **Tag Resources for Easy Identification**
```hcl
resource "azurerm_linux_virtual_machine" "app" {
  name = "app-vm"
  # ... other config
  
  tags = {
    turbonomic_id    = "vm-12345"           # Turbonomic entity ID
    terraform_var    = "usr_vm_size"        # Variable controlling size
    cost_center      = "engineering"
    managed_by       = "terraform"
  }
}
```

### 3. **Create a Variables Mapping File**
Create a `turbonomic-mapping.yaml` in your repo:
```yaml
resources:
  - turbonomic_id: "vm-12345"
    resource_type: "azurerm_linux_virtual_machine"
    resource_name: "azurerm_linux_virtual_machine.app"
    mappings:
      instanceType_after: "var.usr_vm_size"
      storageAmount_after: "var.usr_os_disk_size"
    
  - turbonomic_id: "db-67890"
    resource_type: "azurerm_mssql_database"
    resource_name: "azurerm_mssql_database.main"
    mappings:
      instanceType_after: "var.usr_db_sku"
      storageAmount_after: "var.usr_db_storage"
```

### 4. **Automate with Scripts**
Use the Turbonomic API to export recommendations and match them to Terraform:
```bash
# Export Turbonomic recommendations
curl -X GET "https://turbonomic-instance/api/v3/markets/Market/actions" \
  -H "Authorization: Bearer $TOKEN" > recommendations.json

# Parse and match to Terraform variables
python3 map-recommendations.py
```

### 5. **Version Control Your Mappings**
Keep mapping documentation in your Terraform repository alongside your code so teams can reference it during reviews and updates.

---

## Additional Resources

- **Turbonomic API Documentation**: Use the Actions API to programmatically retrieve recommendations
- **Terraform State**: `terraform state` commands help identify current resource configurations
- **Azure Resource SKUs**: Reference [Azure VM sizes](https://learn.microsoft.com/azure/virtual-machines/sizes) and other SKU documentation
- **Automation**: Consider CI/CD integration to automatically create PRs with Turbonomic recommendations

---

## Notes

1. **Custom Variable Names**: Your organization may use different variable naming conventions. This guide shows common patterns but adjust based on your standards.

2. **Resource Addressing**: Turbonomic uses Azure Resource IDs; cross-reference these with `terraform state` output.

3. **SKU Formats**: Azure SKU naming varies by service. Always verify the exact format in Azure documentation.

4. **Autoscaling**: For autoscale-enabled resources, map to `min_count`/`max_count` rather than fixed `count`.

5. **Multi-Region**: If deploying across regions, ensure mappings account for region-specific SKU availability.



