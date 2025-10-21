# synapse_mcp_client.py
import asyncio
from fastmcp import Client

async def main():
    # Connect to the MCP server (HTTP transport example)
    client = await Client.connect("http://localhost:8000/mcp")

    # 1. List tables
    tables = await client.call_tool("list_tables", {"schema": "dbo"})
    print("Tables:", tables)

    # 2. Describe a table
    desc = await client.call_tool("describe_table", {"table_name": "dbo.MyTable"})
    print("Schema:", desc)

    # 3. Run parameterized query
    result = await client.call_tool("run_query", {
        "sql": "SELECT * FROM dbo.CostSummary WHERE UsageDate >= ?",
        "parameters": ["2025-10-01"],
        "limit": 10
    })
    print("Query result:", result)

    # 4. Execute stored procedure
    sp = await client.call_tool("exec_stored_procedure", {
        "procedure_name": "dbo.usp_RefreshCostSnapshot",
        "parameters": []
    })
    print("SP result:", sp)

    # 5. Get summary
    summary = await client.call_tool("get_summary", {
        "table": "dbo.CostSummary",
        "column": "CostAmount",
        "agg_func": "SUM"
    })
    print("Summary:", summary)

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
