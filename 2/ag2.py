# langchain_router_agent.py

import json
import asyncio
from langchain.agents import Tool
from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from mcp_tool_loader import load_mcp_tools


# ───────────────────────────────────────────────
# 1. Build Sub-Agents from MCP tools
# ───────────────────────────────────────────────

async def build_subagents():
    mcp_tools = await load_mcp_tools()

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # Identify tools dynamically
    yaml_tool = next(t for t in mcp_tools if t.name == "yaml_updater_tool")
    locals_tool = next(t for t in mcp_tools if t.name == "locals_updater_tool")

    # Build sub-agents using create_agent()
    yaml_agent = create_agent(
        llm=llm,
        tools=[yaml_tool],
        state_modifier="You are the YAML updater agent. Only call yaml_updater_tool."
    )

    locals_agent = create_agent(
        llm=llm,
        tools=[locals_tool],
        state_modifier="You are the LOCALS updater agent. Only call locals_updater_tool."
    )

    return yaml_agent, locals_agent, mcp_tools


# ───────────────────────────────────────────────
# 2. Router Agent
# ───────────────────────────────────────────────

def build_router_agent(llm):
    router_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
Decide which tool to call.

Rules:
- If file contains 'locals {' or is terraform/HCL → return "locals_updater_tool"
- If file ends with .yaml or .yml → return "yaml_updater_tool"

Return ONLY the tool name.
Text:
{input}
"""
    )

    router_agent = create_agent(
        llm=llm,
        tools=[],  # Router does NOT execute tools directly
        prompt=router_prompt,
    )

    return router_agent


# ───────────────────────────────────────────────
# 3. Parent Orchestrator Agent
# ───────────────────────────────────────────────

async def build_parent_agent():
    yaml_agent, locals_agent, all_mcp_tools = await build_subagents()

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    router = build_router_agent(llm)

    async def parent_executor(user_input: str):
        """Parent agent pipeline"""

        # Step 1: Router decides
        tool_choice = await router.ainvoke({"input": user_input})
        chosen_tool = tool_choice["output"].strip()

        # Step 2: Parse arguments
        extract_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
Extract JSON arguments required for the tool call.

Return ONLY a valid JSON object with:
- terraform_file
- metadata_file
- updates

User text:
{input}
"""
        )

        extractor = create_agent(llm=llm, tools=[], prompt=extract_prompt)
        parsed = await extractor.ainvoke({"input": user_input})
        args = json.loads(parsed["output"])

        # Step 3: Run correct subagent
        if chosen_tool == "yaml_updater_tool":
            return await yaml_agent.ainvoke(json.dumps(args))

        if chosen_tool == "locals_updater_tool":
            return await locals_agent.ainvoke(json.dumps(args))

        return f"Unknown tool: {chosen_tool}"

    return parent_executor


# ───────────────────────────────────────────────
# 4. Demo
# ───────────────────────────────────────────────

async def main():
    parent = await build_parent_agent()

    query = """
Update locals.tf using metadata.json.
Updates: [{"env":"dev","name":"sql","value":50}]
terraform_file="locals.tf"
metadata_file="metadata.json"
"""

    result = await parent(query)
    print("\nFINAL RESULT\n", result)


if __name__ == "__main__":
    asyncio.run(main())
