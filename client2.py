from mcp.client import MCPClient

# MCP server endpoint
MCP_URL = "http://localhost:8000/mcp"

# Initialize client
client = MCPClient(MCP_URL)


# 1️⃣ List all tables in default schema
print("\n=== List Tables ===")
tables = client.call_tool("list_tables", {})
print(tables)


# 2️⃣ Describe a specific table (e.g., dbo.Customers)
print("\n=== Describe Table ===")
desc = client.call_tool("describe_table", {"table_name": "dbo.Customers"})
print(desc)


# 3️⃣ Run a parameterized query (example: filter by region)
print("\n=== Run Parameterized Query ===")
query_result = client.call_tool("run_query", {
    "sql": "SELECT CustomerID, CustomerName, Region FROM dbo.Customers WHERE Region = ?",
    "parameters": ["West"],
    "limit": 5
})
print(query_result)


# 4️⃣ Run another query (different table and condition)
print("\n=== Run Another Query (Orders table) ===")
orders_result = client.call_tool("run_query", {
    "sql": "SELECT TOP 10 OrderID, Amount FROM dbo.Orders WHERE Amount > ?",
    "parameters": [500]
})
print(orders_result)


# 5️⃣ Execute a stored procedure with parameters
print("\n=== Execute Stored Procedure ===")
sp_result = client.call_tool("exec_stored_procedure", {
    "procedure_name": "dbo.usp_GetTopCustomers",
    "parameters": [2025]  # e.g., pass a year
})
print(sp_result)


# 6️⃣ Get summary (aggregation)
print("\n=== Get Aggregation ===")
summary_result = client.call_tool("get_summary", {
    "table": "dbo.Orders",
    "column": "Amount",
    "agg_func": "AVG"
})
print(summary_result)
