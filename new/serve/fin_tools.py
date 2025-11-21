# finops_tools.py
import os
import pyodbc
from typing import List, Any
from pydantic import BaseModel

from fastmcp import FastMCP, Context

# ----------------------------------------
# FinOps initialization
# ----------------------------------------
SYNAPSE_CONNSTR = os.getenv("SYNAPSE_CONNECT_STRING")
if not SYNAPSE_CONNSTR:
    raise RuntimeError("Missing SYNAPSE_CONNECT_STRING in environment.")

def get_synapse_conn():
    return pyodbc.connect(SYNAPSE_CONNSTR, autocommit=True)

# Exported MCP instance
mcp = FastMCP("finops_tools")

# ----------------------------------------
# FinOps Query Tool
# ----------------------------------------

class FinOpsQueryInput(BaseModel):
    resource_type: str
    region: str
    limit: int = 20

class FinOpsQueryOutput(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


@mcp.tool(name="get_rightsizing_recommendations", description="Query Synapse FinOps DB for right-sizing suggestions")
def get_rightsizing_recommendations(params: FinOpsQueryInput, ctx: Context) -> FinOpsQueryOutput:
    conn = get_synapse_conn()
    cur = conn.cursor()

    query = f"""
        SELECT TOP {params.limit}
        *
        FROM finops_table
        WHERE resource_type = ? AND region = ?
    """
    cur.execute(query, (params.resource_type, params.region))
    rows = [list(r) for r in cur.fetchall()]
    cols = [c[0] for c in cur.description]

    conn.close()
    return FinOpsQueryOutput(columns=cols, rows=rows)
