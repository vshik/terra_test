import os
import pyodbc
from typing import List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.transports.http import HTTPTransport

# Load env vars
load_dotenv()

# Initialize FastMCP service
mcp = FastMCP("synapse_mcp")

CONNSTR = os.getenv("SYNAPSE_CONNECT_STRING")
if not CONNSTR:
    raise RuntimeError("Missing SYNAPSE_CONNECT_STRING in environment.")


def get_conn():
    return pyodbc.connect(CONNSTR, autocommit=True)


# ---- 1️. List Tables ----
class ListInput(BaseModel):
    schema: Optional[str] = None

class ListOutput(BaseModel):
    tables: List[str]

@mcp.tool(name="list_tables", description="List tables in Synapse (optionally by schema)")
def list_tables(params: ListInput, ctx: Context) -> ListOutput:
    conn = get_conn()
    cur = conn.cursor()
    if params.schema:
        cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?", params.schema)
    else:
        cur.execute("SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return ListOutput(tables=tables)


# ---- 2️. Describe Table ----
class DescribeInput(BaseModel):
    table_name: str

class DescribeOutput(BaseModel):
    columns: List[str]

@mcp.tool(name="describe_table", description="Describe table columns")
def describe_table(params: DescribeInput, ctx: Context) -> DescribeOutput:
    conn = get_conn()
    cur = conn.cursor()
    if "." in params.table_name:
        schema, table = params.table_name.split(".", 1)
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
        """, params.table_name)
    cols = [r[0] for r in cur.fetchall()]
    conn.close()
    return DescribeOutput(columns=cols)


# ---- 3️. Run Query ----
class QueryInput(BaseModel):
    sql: str
    parameters: Optional[List[Any]] = None
    limit: int = 100

class QueryOutput(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

@mcp.tool(name="run_query", description="Run a SELECT query (parameterized)")
def run_query(params: QueryInput, ctx: Context) -> QueryOutput:
    sql = params.sql.strip()
    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT statements allowed.")
    conn = get_conn()
    cur = conn.cursor()
    safe_sql = f"SELECT TOP {params.limit} * FROM ({params.sql}) AS sub"
    cur.execute(safe_sql, params.parameters or [])
    cols = [c[0] for c in cur.description]
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return QueryOutput(columns=cols, rows=rows)


# ---- 4️. Stored Procedure ----
class SPInput(BaseModel):
    procedure_name: str
    parameters: Optional[List[Any]] = None

class SPOutput(BaseModel):
    result: List[List[Any]]

@mcp.tool(name="exec_stored_procedure", description="Execute a stored procedure")
def exec_stored_procedure(params: SPInput, ctx: Context) -> SPOutput:
    conn = get_conn()
    cur = conn.cursor()
    if params.parameters:
        placeholders = ",".join("?" for _ in params.parameters)
        sql = f"EXEC {params.procedure_name} {placeholders}"
        cur.execute(sql, params.parameters)
    else:
        sql = f"EXEC {params.procedure_name}"
        cur.execute(sql)
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return SPOutput(result=rows)


# ---- 5️. Summary ----
class SummaryInput(BaseModel):
    table: str
    column: str
    agg_func: str = Field(default="AVG")

class SummaryOutput(BaseModel):
    summary: Any

@mcp.tool(name="get_summary", description="Aggregate a column (SUM, AVG, COUNT, etc.)")
def get_summary(params: SummaryInput, ctx: Context) -> SummaryOutput:
    conn = get_conn()
    cur = conn.cursor()
    query = f"SELECT {params.agg_func}([{params.column}]) FROM {params.table}"
    cur.execute(query)
    val = cur.fetchone()[0]
    conn.close()
    return SummaryOutput(summary=val)


# ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("MCP_PORT", 8000))

    transport = HTTPTransport(host="0.0.0.0", port=PORT, path="/mcp")
    # This launches a uvicorn server automatically
    mcp.run(transport)
