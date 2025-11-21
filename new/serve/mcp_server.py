# mcp_server.py
import os
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment early
load_dotenv()

# Import the two MCP tool modules
from github_tools import mcp as github_mcp
from finops_tools import mcp as finops_mcp

# Create a unified composite MCP server
server = FastMCP("finops_github_mcp")

# Merge tool registries
server.tools.update(github_mcp.tools)
server.tools.update(finops_mcp.tools)

if __name__ == "__main__":
    PORT = int(os.getenv("MCP_PORT", 8100))
    server.run(transport="http", host="0.0.0.0", port=PORT)
