import os
import pyodbc
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Azure Synapse FinOps MCP Server")
mcp = FastMCP("finops_mcp")

CONNSTR = os.getenv("SYNAPSE_CONNSTRING")
PORT = int(os.getenv("MCP_PORT", 8000))
SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")

def get_conn():
    if not CONNSTR:
        raise RuntimeError("Missing SYNAPSE_CONNSTRING in environment.")
    return pyodbc.connect(CONNSTR, autocommit=True)

def deploy_sql_scripts():
    """Deploy all .sql files in the /sql directory into Synapse."""
    conn = get_conn()
    cursor = conn.cursor()

    for file in os.listdir(SQL_DIR):
        if file.endswith(".sql"):
            path = os.path.join(SQL_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                sql_script = f.read().strip()
                print(f"Deploying {file}...")
                try:
                    cursor.execute(sql_script)
                    conn.commit()
                    print(f"{file} deployed successfully.")
                except Exception as e:
                    print(f"Error deploying {file}: {e}")

    cursor.close()
    conn.close()

# ---------- Input/Output Models ----------
class SPInput(BaseModel):
    procedure_name: str
    parameters: Optional[List[Any]] = None

class SPOutput(BaseModel):
    result: List[List[Any]]


# ---------- Generic Stored Procedure Executor ----------
@mcp.tool(name="run_stored_procedure", description="Execute a stored procedure in Synapse")
def run_stored_procedure(params: SPInput) -> SPOutput:
    conn = get_conn()
    cur = conn.cursor()
    query = f"EXEC {params.procedure_name}"
    if params.parameters:
        placeholders = ",".join("?" for _ in params.parameters)
        query = f"{query} {placeholders}"
        cur.execute(query, params.parameters)
    else:
        cur.execute(query)
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return SPOutput(result=rows)


# ---------- Turbonomic + FinOps Stored Procedure Wrappers ----------

# Example wrapper for easier LLM use — these just call run_stored_procedure()
FINOPS_PROCS = [
    "usp_GetTotalCloudCost",
    "usp_GetCostTrendByService",
    "usp_GetTopResourceGroupsByCost",
    "usp_GetTopResourcesByCost",
    "usp_GetMonthlyCostBySubscription",
    "usp_GetRightSizingRecommendations",
    "usp_GetCPUUtilizationSummary",
    "usp_GetMemoryUtilizationSummary",
    "usp_GetIdleResourceList",
    "usp_GetUnattachedDiskList",
    "usp_GetSavingsOpportunitiesByService",
    "usp_GetAnomalyCostEvents",
    "usp_GetTagComplianceSummary",
    "usp_GetBudgetVsActual",
    "usp_GetStorageOptimizationSummary"
]


for proc_name in FINOPS_PROCS:
    @mcp.tool(name=proc_name, description=f"Execute the FinOps stored procedure {proc_name}")
    def _wrapper(params: Optional[List[Any]] = None, proc=proc_name):
        conn = get_conn()
        cur = conn.cursor()
        query = f"EXEC {proc}"
        if params:
            placeholders = ",".join("?" for _ in params)
            query = f"{query} {placeholders}"
            cur.execute(query, params)
        else:
            cur.execute(query)
        rows = [list(r) for r in cur.fetchall()]
        conn.close()
        return SPOutput(result=rows)


@app.tool()
def list_stored_procedures():
    """List all registered FinOps stored procedures."""
    return {"procedures": FINOPS_PROCS}

@app.tool()
def run_stored_procedure(proc_name: str):
    """Run one of the predefined FinOps stored procedures."""
    if proc_name not in FINOPS_PROCS:
        return {"error": f"Stored procedure '{proc_name}' is not registered."}

    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(f"EXEC {proc_name}")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        return {"data": result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

@app.get("/health")
def health():
    return {"status": "ok"}


# Mount MCP app
app.mount("/mcp", mcp.as_fastapi())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
