import asyncio
from fastmcp import Client

MCP_URL = "http://localhost:8000/mcp"


async def main():
    async with Client(MCP_URL) as client:
        # Example 1: Get total cloud cost
        total_cost = await client.call_tool(
            "usp_GetTotalCloudCost",
            {"params": {}}
        )
        print("Total Cloud Cost:", total_cost)

        # Example 2: Get top resource groups by cost
        top_rg = await client.call_tool(
            "usp_GetTopResourceGroupsByCost",
            {"params": {}}
        )
        print("Top Resource Groups by Cost:", top_rg)

        # Example 3: Get right-sizing recommendations from Turbonomic
        rightsizing = await client.call_tool(
            "usp_GetRightSizingRecommendations",
            {"params": {}}
        )
        print("Right-sizing Recommendations:", rightsizing)

        # Example 4: CPU Utilization Summary
        cpu = await client.call_tool(
            "usp_GetCPUUtilizationSummary",
            {"params": {}}
        )
        print("CPU Utilization Summary:", cpu)

        # Example 5: Get Tag Compliance Summary
        tags = await client.call_tool(
            "usp_GetTagComplianceSummary",
            {"params": {}}
        )
        print("Tag Compliance Summary:", tags)


if __name__ == "__main__":
    asyncio.run(main())
