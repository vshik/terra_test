import requests
import json

MCP_URL = "http://localhost:8000/mcp"

def call_mcp(method, params=None, id=1):
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": id
    }
    resp = requests.post(MCP_URL, json=payload)
    print(f"\n➡️ Request: {json.dumps(payload, indent=2)}")
    print(f"⬅️ Response: {json.dumps(resp.json(), indent=2)}")
    return resp.json()

# --- Example calls ---

call_mcp("list_tables", {"schema": "dbo"})
call_mcp("describe_table", {"table_name": "dbo.Customers"})
call_mcp("run_query", {
    "sql": "SELECT TOP 5 * FROM dbo.Orders WHERE Amount > ?",
    "parameters": [1000]
})
call_mcp("get_summary", {
    "table": "dbo.Orders",
    "column": "Amount",
    "agg_func": "AVG"
})
