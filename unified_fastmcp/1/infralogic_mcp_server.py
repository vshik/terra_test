# mcp/infralogic_mcp/server.py

import os
import json
import asyncio
from typing import Optional
from fastmcp import MCP, Context
from pydantic import BaseModel

# Import your internal functions
from tools.analyze_tf_file import analyze_tf_file
from tools.find_rightsizing_vars import find_rightsizing_vars
from tools.update_tf_file import update_tf_file


# --- Define Input/Output Schemas ---
class AnalyzeInput(BaseModel):
    repo_path: str  # e.g., "c:/repos/repo_123"
    branch: Optional[str] = "main"

class AnalyzeOutput(BaseModel):
    analysis_path: str  # Path to JSON output with TF variable metadata


class UpdateInput(BaseModel):
    repo_path: str
    analysis_path: str  # Path to JSON output from analyze_infra
    recommendations: dict  # FinOps right-sizing recommendations


class UpdateOutput(BaseModel):
    updated_files: list  # List of updated Terraform file paths
    summary: str


# --- Initialize MCP server ---
mcp = MCP(name="InfraLogic MCP", description="Terraform Analysis and Update Tools")


# --- Tool 1: Analyze Infra ---
@mcp.tool(name="analyze_infra", description="Analyze Terraform repo for right-sizing variables")
async def analyze_infra(params: AnalyzeInput, ctx: Context) -> AnalyzeOutput:
    """Scan Terraform repo for right-sizing variable locations."""
    repo_path = params.repo_path

    # Step 1: Analyze Terraform files
    tf_analysis = await asyncio.to_thread(analyze_tf_file, repo_path)

    # Step 2: Extract right-sizing vars
    rightsizing_info = await asyncio.to_thread(find_rightsizing_vars, tf_analysis)

    # Step 3: Save JSON result
    output_file = os.path.join(repo_path, "tf_rightsizing_map.json")
    with open(output_file, "w") as f:
        json.dump(rightsizing_info, f, indent=2)

    return AnalyzeOutput(analysis_path=output_file)


# --- Tool 2: Update Infra ---
@mcp.tool(name="update_infra", description="Update Terraform files based on recommendations")
async def update_infra(params: UpdateInput, ctx: Context) -> UpdateOutput:
    """Update Terraform files with new right-sizing variable values."""
    repo_path = params.repo_path
    analysis_path = params.analysis_path
    recommendations = params.recommendations

    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)

    # Update TF files based on analysis + recommendations
    updated_files = await asyncio.to_thread(update_tf_file, repo_path, analysis_data, recommendations)

    summary_msg = f" Updated {len(updated_files)} Terraform files in {repo_path}"
    return UpdateOutput(updated_files=updated_files, summary=summary_msg)


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    mcp.run(host="127.0.0.1", port=9500)
