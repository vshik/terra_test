# synapse_mcp_server.py
import os
import pyodbc
from typing import List, Any, Optional, Dict
from fastmcp import FastMCP, Context   # Requires fastmcp v2.x  :contentReference[oaicite:3]{index=3}
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Synapse FinOps Service")

CONNSTR = os.getenv("SYNAPSE_CONNSTRING")
if not CONNSTR:
    raise RuntimeError("Missing environment variable SYNAPSE_CONNSTRING")

def get_conn():
    return pyodbc.connect(CONNSTR, autocommit=True)

# Input/output schemas
class ListTablesParams(BaseModel):
    schema: Optional[str] = Field(None, description="Optional schema name to filter")

class ListTablesResult(BaseModel):
    tables: List[str]

class DescribeTableParams(BaseModel):
    table_name: str = Field(..., description="Fully qualified table name (schema.table) or table")

class DescribeTableResult(BaseModel):
    columns: List[str]

class RunQueryParams(BaseModel):
    sql: str = Field(..., description="SELECT query; should be parameterized")
    parameters: Optional[List[Any]] = Field(default=None, description="List of parameters matching '?' markers")
    limit: int = Field(default=100, description="Max rows to return")

class RunQueryResult(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

class ExecProcParams(BaseModel):
    procedure_name: str = Field(..., description="Stored procedure name (with schema)")
    parameters: Optional[List[Any]] = Field(default=None, description="Parameters for the procedure")

class ExecProcResult(BaseModel):
    result: List[List[Any]]

class GetSummaryParams(BaseModel):
    table: str = Field(..., description="Table name (schema.table or table)")
    column: str = Field(..., description="Column for aggregation")
    agg_func: str = Field(default="AVG", description="Aggregate function: SUM, AVG, COUNT, MAX, MIN")

class GetSummaryResult(BaseModel):
    summary: Any

# Tool implementations
@mcp.tool(name="list_tables", description="List tables in Synapse (optionally filtered by schema)")
def list_tables(params: ListTablesParams, ctx: Context) -> ListTablesResult:
    conn = get_conn()
    cursor = conn.cursor()
    if params.schema:
        cursor.execute(
            "SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?",
            params.schema
        )
    else:
        cursor.execute("SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")
    rows = [r[0] for r in cursor.fetchall()]
    conn.close()
    return ListTablesResult(tables=rows)

@mcp.tool(name="describe_table", description="Describe the schema of a given table (columns + data types)")
def describe_table(params: DescribeTableParams, ctx: Context) -> DescribeTableResult:
    conn = get_conn()
    cursor = conn.cursor()
    if "." in params.table_name:
        schema, tbl = params.table_name.split(".", 1)
        cursor.execute(
            "SELECT COLUMN_NAME + ' ' + DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            schema, tbl
        )
    else:
        cursor.execute(
            "SELECT COLUMN_NAME + ' ' + DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            params.table_name
        )
    cols = [r[0] for r in cursor.fetchall()]
    conn.close()
    return DescribeTableResult(columns=cols)

@mcp.tool(name="run_query", description="Run a parameterized SELECT query (limited rows)")
def run_query(params: RunQueryParams, ctx: Context) -> RunQueryResult:
    sql = params.sql.strip()
    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed")
    conn = get_conn()
    cursor = conn.cursor()
    wrapped_sql = f"SELECT TOP {params.limit} * FROM ({sql}) AS sub_query"
    if params.parameters:
        cursor.execute(wrapped_sql, *params.parameters)
    else:
        cursor.execute(wrapped_sql)
    columns = [c[0] for c in cursor.description]
    rows = [list(r) for r in cursor.fetchall()]
    conn.close()
    return RunQueryResult(columns=columns, rows=rows)

@mcp.tool(name="exec_stored_procedure", description="Execute a stored procedure with optional parameters")
def exec_stored_procedure(params: ExecProcParams, ctx: Context) -> ExecProcResult:
    conn = get_conn()
    cursor = conn.cursor()
    if params.parameters:
        placeholders = ",".join("?" for _ in params.parameters)
        sql = f"EXEC {params.procedure_name} {placeholders}"
        cursor.execute(sql, *params.parameters)
    else:
        sql = f"EXEC {params.procedure_name}"
        cursor.execute(sql)
    rows = [list(r) for r in cursor.fetchall()]
    conn.close()
    return ExecProcResult(result=rows)

@mcp.tool(name="get_summary", description="Return an aggregated value (SUM, AVG, COUNT, etc.) on a column")
def get_summary(params: GetSummaryParams, ctx: Context) -> GetSummaryResult:
    conn = get_conn()
    cursor = conn.cursor()
    safe_sql = f"SELECT {params.agg_func}([{params.column}]) FROM {params.table}"
    cursor.execute(safe_sql)
    val = cursor.fetchone()[0]
    conn.close()
    return GetSummaryResult(summary=val)

if __name__ == "__main__":
    # Run server using fastmcp default transport (stdio) or specify HTTP/SSE
    mcp.run(transport="http", host="0.0.0.0", port=8000, path="/mcp")  # example HTTP transport
