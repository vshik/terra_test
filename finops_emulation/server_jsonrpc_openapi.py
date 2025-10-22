import os
import pyodbc
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Any, Optional, Dict, Callable

load_dotenv()
app = FastAPI(title="Azure Synapse MCP JSON-RPC + OpenAPI Server")

CONNSTR = os.getenv("SYNAPSE_CONNSTRING")
PORT = int(os.getenv("MCP_PORT", 8000))


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def get_conn():
    if not CONNSTR:
        raise RuntimeError("Missing SYNAPSE_CONNSTRING in environment")
    return pyodbc.connect(CONNSTR, autocommit=True)


# ----------------------------------------------------------------------
# Tool Registry: Define schemas, functions, and metadata
# ----------------------------------------------------------------------
class ListInput(BaseModel):
    schema: Optional[str] = Field(None, description="Schema name (optional)")

class ListOutput(BaseModel):
    tables: List[str]

class DescribeInput(BaseModel):
    table_name: str = Field(..., description="Fully qualified table name (schema.table)")

class DescribeOutput(BaseModel):
    columns: List[str]

class QueryInput(BaseModel):
    sql: str = Field(..., description="SQL SELECT query")
    parameters: Optional[List[Any]] = Field(default=None, description="Query parameters")
    limit: int = Field(default=100, description="Row limit")

class QueryOutput(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

class SPInput(BaseModel):
    procedure_name: str
    parameters: Optional[List[Any]] = None

class SPOutput(BaseModel):
    result: List[List[Any]]

class SummaryInput(BaseModel):
    table: str
    column: str
    agg_func: str = Field(default="AVG", description="Aggregate function: SUM, AVG, COUNT, etc.")

class SummaryOutput(BaseModel):
    summary: Any


# ----------------------------------------------------------------------
# Core tool functions
# ----------------------------------------------------------------------
def list_tables(params: Dict[str, Any]) -> Dict[str, Any]:
    schema = params.get("schema")
    conn = get_conn()
    cur = conn.cursor()
    if schema:
        cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?", schema)
    else:
        cur.execute("SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")
    result = [r[0] for r in cur.fetchall()]
    conn.close()
    return {"tables": result}


def describe_table(params: Dict[str, Any]) -> Dict[str, Any]:
    table_name = params["table_name"]
    conn = get_conn()
    cur = conn.cursor()
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        cur.execute("""
            SELECT COLUMN_NAME + ' ' + DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, schema, table)
    else:
        cur.execute("""
            SELECT COLUMN_NAME + ' ' + DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, table_name)
    result = [r[0] for r in cur.fetchall()]
    conn.close()
    return {"columns": result}


def run_query(params: Dict[str, Any]) -> Dict[str, Any]:
    sql = params["sql"]
    parameters = params.get("parameters", [])
    limit = params.get("limit", 100)
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements allowed")
    conn = get_conn()
    cur = conn.cursor()
    safe_sql = f"SELECT TOP {limit} * FROM ({sql}) AS sub"
    cur.execute(safe_sql, parameters)
    cols = [c[0] for c in cur.description]
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return {"columns": cols, "rows": rows}


def exec_stored_procedure(params: Dict[str, Any]) -> Dict[str, Any]:
    proc = params["procedure_name"]
    parameters = params.get("parameters", [])
    conn = get_conn()
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in parameters)
    query = f"EXEC {proc} {placeholders}" if placeholders else f"EXEC {proc}"
    cur.execute(query, parameters)
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return {"result": rows}


def get_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    table = params["table"]
    column = params["column"]
    agg = params.get("agg_func", "AVG")
    conn = get_conn()
    cur = conn.cursor()
    query = f"SELECT {agg}([{column}]) FROM {table}"
    cur.execute(query)
    val = cur.fetchone()[0]
    conn.close()
    return {"summary": val}


# ----------------------------------------------------------------------
# Tool registry
# ----------------------------------------------------------------------
TOOLS: Dict[str, Dict[str, Any]] = {
    "list_tables": {
        "func": list_tables,
        "description": "List all tables in Synapse (optionally filter by schema)",
        "input_schema": ListInput,
        "output_schema": ListOutput,
    },
    "describe_table": {
        "func": describe_table,
        "description": "Describe table columns and types",
        "input_schema": DescribeInput,
        "output_schema": DescribeOutput,
    },
    "run_query": {
        "func": run_query,
        "description": "Run a safe parameterized SELECT query (limited rows)",
        "input_schema": QueryInput,
        "output_schema": QueryOutput,
    },
    "exec_stored_procedure": {
        "func": exec_stored_procedure,
        "description": "Execute a stored procedure with optional parameters",
        "input_schema": SPInput,
        "output_schema": SPOutput,
    },
    "get_summary": {
        "func": get_summary,
        "description": "Return a summary aggregation (SUM, AVG, COUNT...)",
        "input_schema": SummaryInput,
        "output_schema": SummaryOutput,
    },
}


# ----------------------------------------------------------------------
# JSON-RPC endpoint
# ----------------------------------------------------------------------
@app.post("/mcp")
async def mcp_entrypoint(req: Request):
    data = await req.json()
    method = data.get("method")
    params = data.get("params", {})
    request_id = data.get("id")

    if method not in TOOLS:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"},
        }

    tool = TOOLS[method]
    try:
        # Validate input using the tool's Pydantic schema
        validated_params = tool["input_schema"](**params).dict()
        result = tool["func"](validated_params)
        # Validate output schema before returning
        validated_result = tool["output_schema"](**result).dict()
        return {"jsonrpc": "2.0", "id": request_id, "result": validated_result}
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(e)},
        }


# ----------------------------------------------------------------------
# OpenAPI Metadata endpoints
# ----------------------------------------------------------------------
@app.get("/openapi-tools")
def list_tools():
    """
    List all available MCP tools with descriptions.
    """
    return [
        {
            "name": name,
            "description": meta["description"],
            "input_schema": meta["input_schema"].schema(),
            "output_schema": meta["output_schema"].schema(),
        }
        for name, meta in TOOLS.items()
    ]


@app.get("/openapi-tools/{tool_name}")
def get_tool_schema(tool_name: str):
    """
    Get full OpenAPI schema for one tool.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="Tool not found")
    meta = TOOLS[tool_name]
    return {
        "name": tool_name,
        "description": meta["description"],
        "input_schema": meta["input_schema"].schema(),
        "output_schema": meta["output_schema"].schema(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_jsonrpc_openapi:app", host="0.0.0.0", port=PORT, reload=True)
