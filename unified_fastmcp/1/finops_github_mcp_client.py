import asyncio
import os
from dotenv import load_dotenv
from fastmcp import AsyncClient

load_dotenv()

MCP_URL = os.getenv("FINOPS_GITHUB_MCP_URL", "http://localhost:8100/mcp")

async def main():
    async with AsyncClient(MCP_URL) as client:
        # ---------- GITHUB WORKFLOW ----------
        repo_url = "https://github.com/your-org/terraform-repo.git"
        repo_fullname = "your-org/terraform-repo"
        feature_branch = "rightsizing-update"

        # Clone repo
        clone = await client.call_tool("clone_repo", {"repo_url": repo_url, "branch": "main"})
        local_path = clone["local_path"]
        print("Repo cloned to:", local_path)

        # Create branch
        await client.call_tool("create_branch", {
            "repo_path": local_path,
            "branch_name": feature_branch,
            "base_branch": "main"
        })
        print(f"Created and switched to branch '{feature_branch}'")

        # (Optional) modify Terraform vars here...

        # Commit & push
        await client.call_tool("commit_and_push", {
            "repo_path": local_path,
            "branch": feature_branch,
            "commit_message": "Automated right-sizing update"
        })
        print("Pushed commit")

        # Create PR
        pr = await client.call_tool("create_pull_request", {
            "repo_fullname": repo_fullname,
            "source_branch": feature_branch,
            "target_branch": "main",
            "title": "Automated Right-Sizing Update",
            "body": "Updated Terraform variables based on FinOps recommendations."
        })
        print(f"PR Created: {pr['pr_url']}")

        # ---------- FINOPS QUERY ----------
        finops_resp = await client.call_tool("get_rightsizing_recommendations", {
            "resource_type": "vm",
            "region": "eastus2"
        })
        print("\nFinOps Recommendations:")
        for r in finops_resp["recommendations"]:
            print(f" - {r['resource_type']} in {r['region']}: {r['current_size']} → {r['recommended_size']} (${r['monthly_saving_usd']}/mo)")

if __name__ == "__main__":
    asyncio.run(main())
