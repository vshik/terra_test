# server.py
import uvicorn
from fastapi import FastAPI
from mcp.server.fastapi import MCPServer

# Your existing update functions
from your_file import (
    update_terraform_file_locals_type,
    update_terraform_file_yaml_type
)

app = FastAPI()
server = MCPServer(app)

# -------------------------------------------------
# MCP Tools
# -------------------------------------------------

@server.tool()
async def locals_updater(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform LOCALS file."""
    return update_terraform_file_locals_type(terraform_file, metadata_file, updates, similarity_threshold)


@server.tool()
async def yaml_updater(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """Update a Terraform YAML file."""
    return update_terraform_file_yaml_type(terraform_file, metadata_file, updates, similarity_threshold)


# -------------------------------------------------
# Start HTTP-based MCP server
# -------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
