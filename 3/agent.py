# agent.py
import asyncio
import json
from mcp.client import Client
from mcp.client.http import http_client

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------
# 1. LLM
# -------------------------------------------------
llm = ChatOpenAI(model="gpt-4.1", temperature=0)


# -------------------------------------------------
# 2. Router Prompt
# -------------------------------------------------
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

router_chain = ROUTER_PROMPT | llm | StrOutputParser()


# -------------------------------------------------
# 3. Argument JSON extractor
# -------------------------------------------------
PARSE_PROMPT = PromptTemplate(
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

parse_chain = PARSE_PROMPT | llm | StrOutputParser()


# -------------------------------------------------
# 4. Router Agent
# -------------------------------------------------
async def router_agent(user_input: str):
    decision = router_chain.invoke({"input": user_input}).strip()
    print(f"\n🔍 Router decision → {decision}\n")

    # Extract arguments
    parsed_json = parse_chain.invoke({"input": user_input})
    args = json.loads(parsed_json)

    # Connect to MCP server via HTTP
    async with http_client("http://localhost:8001/mcp") as (read, write):
        client = Client(read, write)
        discovery = await client.list_tools()
        tool_names = {t.name for t in discovery}

        if decision not in tool_names:
            return f"Router error: unknown tool '{decision}'"

        result = await client.call_tool(decision, args)
        return result


# -------------------------------------------------
# 5. Example main()
# -------------------------------------------------
async def main():
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
    output = await router_agent(query)
    print("\n=== Final Output ===\n")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
