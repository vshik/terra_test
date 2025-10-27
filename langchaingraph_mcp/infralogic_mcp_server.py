from fastmcp import FastMCP, Context
from pydantic import BaseModel
import json
from tools.analyze_tf_file import analyze_tf_file
from tools.update_tf_file import update_tf_file

mcp = FastMCP(name="InfraLogic MCP")

class AnalyzeInput(BaseModel):
    repo_path: str

class AnalyzeOutput(BaseModel):
    analysis_json: str

@mcp.tool(name="analyze_tf_file", description="Analyze Terraform repo")
def analyze_tf_file_tool(params: AnalyzeInput, ctx: Context) -> AnalyzeOutput:
    analysis_path = analyze_tf_file(params.repo_path)
    return AnalyzeOutput(analysis_json=analysis_path)

class UpdateInput(BaseModel):
    repo_path: str
    analysis_json: str
    recommendations: dict

@mcp.tool(name="update_tf_file", description="Update Terraform files")
def update_tf_file_tool(params: UpdateInput, ctx: Context):
    return update_tf_file(params.repo_path, params.analysis_json, params.recommendations)

if __name__ == "__main__":
    mcp.run(port=9500)
