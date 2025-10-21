import asyncio
import os
from fastmcp import AsyncClient
from dotenv import load_dotenv

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")


async def main():
    # Initialize async FastMCP client
    async with AsyncClient(MCP_URL) as client:
        # 1️. List Tables
        tables = await client.call_tool("list_tables", {"schema": "dbo"})
        print("Tables:", tables)

        # 2️. Describe Table
        desc = await client.call_tool("describe_table", {"table_name": "dbo.MyTable"})
        print("Columns:", desc)

        # 3️. Run Parameterized Query
        query = await client.call_tool("run_query", {
            "sql": "SELECT * FROM dbo.MyTable WHERE id > ?",
            "parameters": [100],
            "limit": 10
        })
        print("Query Results:", query)

        # 4️. Execute Stored Procedure
        sp = await client.call_tool("exec_stored_procedure", {
            "procedure_name": "dbo.usp_GetRecentOrders",
            "parameters": [30]
        })
        print("Stored Procedure Output:", sp)

        # 5️. Summary / Aggregation
        summary = await client.call_tool("get_summary", {
            "table": "dbo.Sales",
            "column": "Amount",
            "agg_func": "AVG"
        })
        print("Average Sales:", summary)


if __name__ == "__main__":
    asyncio.run(main())
