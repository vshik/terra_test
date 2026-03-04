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




I'm sharing a Python codebase with multiple sub-directories. Please analyze it thoroughly and produce a structured technical document that covers:

1. **Project Overview**
   - What does this project do? (2-3 sentence summary)
   - What problem does it solve?
   - Key technologies, frameworks, and libraries used

2. **Architecture & Structure**
   - Directory/folder breakdown — what each folder is responsible for
   - High-level architecture diagram (as ASCII or Mermaid)
   - Design patterns used (e.g., MVC, pipeline, microservices)

3. **Core Modules & Components**
   - For each major module/file: purpose, key classes, key functions
   - How modules relate to and depend on each other

4. **Data Flow**
   - How data enters the system (inputs/sources)
   - How it's processed, transformed, and passed between components
   - How it exits the system (outputs/sinks)

5. **Key Functions & Classes**
   - The most important functions/classes and what they do
   - Any critical algorithms or logic worth noting

6. **Configuration & Dependencies**
   - External libraries/packages (from requirements.txt or imports)
   - Environment variables or config files
   - Any external APIs or services integrated

7. **Entry Points**
   - How is the application run/started?
   - CLI commands, main scripts, or API endpoints

8. **Summary & Notes**
   - Any observations about code quality, patterns, or areas of complexity
   - Potential areas to investigate further

Please output this as a well-formatted Word document (.docx) I can share with my team.

