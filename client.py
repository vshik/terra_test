from mcp.client import MCPClient

# MCP endpoint
MCP_URL = "http://localhost:8000/mcp"

client = MCPClient(MCP_URL)

# 1. List tables
tables = client.call_tool("list_tables", {"schema": "dbo"})
print("Tables:", tables)

# 2. Describe table
desc = client.call_tool("describe_table", {"table_name": "dbo.MyTable"})
print("Columns:", desc)

# 3. Run a parameterized query
query = client.call_tool("run_query", {
    "sql": "SELECT * FROM dbo.MyTable WHERE id > ?",
    "parameters": [100],
    "limit": 10
})
print("Query Results:", query)

# 4. Execute stored procedure
sp = client.call_tool("exec_stored_procedure", {
    "procedure_name": "dbo.usp_GetRecentOrders",
    "parameters": [30]
})
print("Stored Procedure Output:", sp)

# 5. Summary / Aggregation
summary = client.call_tool("get_summary", {
    "table": "dbo.Sales",
    "column": "Amount",
    "agg_func": "AVG"
})
print("Average Sales:", summary)
