import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def main():
    transport = StreamableHttpTransport("http://localhost:8000/mcp")
    async with Client(transport) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        # Example 1: List tables
        result = await client.call_tool("list_tables", {"schema": "dbo"})
        print("Tables:", result)

        # Example 2: Query data
        query = "SELECT TOP 5 * FROM dbo.YourTable"
        result = await client.call_tool("run_query", {"sql": query})
        print("Columns:", result["columns"])
        print("Rows:", result["rows"])

if __name__ == "__main__":
    asyncio.run(main())
