import os
import pyodbc
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Azure Synapse MCP Server")
mcp = FastMCP("synapse_mcp")

CONNSTR = os.getenv("SYNAPSE_CONNSTRING")
PORT = int(os.getenv("MCP_PORT", 8000))


def get_conn():
    if not CONNSTR:
        raise RuntimeError("Missing SYNAPSE_CONNSTRING in environment.")
    return pyodbc.connect(CONNSTR, autocommit=True)


# ---- 1️⃣ List Tables/Schemas ----
class ListInput(BaseModel):
    schema: Optional[str] = None

class ListOutput(BaseModel):
    tables: List[str]

@mcp.tool(name="list_tables", description="List tables in Synapse (optionally by schema)")
def list_tables(input: ListInput) -> ListOutput:
    conn = get_conn()
    cur = conn.cursor()
    if input.schema:
        cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?", input.schema)
    else:
        cur.execute("SELECT TABLE_SCHEMA + '.' + TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return ListOutput(tables=tables)


# ---- 2️⃣ Describe Table ----
class DescribeInput(BaseModel):
    table_name: str

class DescribeOutput(BaseModel):
    columns: List[str]

@mcp.tool(name="describe_table", description="Describe table columns with data types")
def describe_table(input: DescribeInput) -> DescribeOutput:
    conn = get_conn()
    cur = conn.cursor()
    if "." in input.table_name:
        schema, table = input.table_name.split(".", 1)
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
        """, input.table_name)
    cols = [r[0] for r in cur.fetchall()]
    conn.close()
    return DescribeOutput(columns=cols)


# ---- 3️⃣ Run Parameterized Query (Safe) ----
class QueryInput(BaseModel):
    sql: str
    parameters: Optional[List[Any]] = None
    limit: int = 100

class QueryOutput(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

@mcp.tool(name="run_query", description="Run a parameterized SELECT query safely (limited rows)")
def run_query(input: QueryInput) -> QueryOutput:
    sql = input.sql.strip().lower()
    if not sql.startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT statements allowed.")

    conn = get_conn()
    cur = conn.cursor()

    safe_sql = f"SELECT TOP {input.limit} * FROM ({input.sql}) as sub"
    cur.execute(safe_sql, input.parameters or [])
    cols = [c[0] for c in cur.description]
    rows = [list(r) for r in cur.fetchall()]

    conn.close()
    return QueryOutput(columns=cols, rows=rows)


# ---- 4️⃣ Execute Stored Procedure ----
class SPInput(BaseModel):
    procedure_name: str
    parameters: Optional[List[Any]] = None

class SPOutput(BaseModel):
    result: List[List[Any]]

@mcp.tool(name="exec_stored_procedure", description="Execute a stored procedure in Synapse")
def exec_stored_procedure(input: SPInput) -> SPOutput:
    conn = get_conn()
    cur = conn.cursor()
    query = f"EXEC {input.procedure_name}"
    if input.parameters:
        placeholders = ",".join("?" for _ in input.parameters)
        query = f"{query} {placeholders}"
        cur.execute(query, input.parameters)
    else:
        cur.execute(query)
    rows = [list(r) for r in cur.fetchall()]
    conn.close()
    return SPOutput(result=rows)


# ---- 5️⃣ Return Aggregation/Summary ----
class SummaryInput(BaseModel):
    table: str
    column: str
    agg_func: str = "AVG"  # SUM, AVG, COUNT, MAX, MIN

class SummaryOutput(BaseModel):
    summary: Any

@mcp.tool(name="get_summary", description="Return aggregated summary (SUM, AVG, COUNT...) of a column")
def get_summary(input: SummaryInput) -> SummaryOutput:
    conn = get_conn()
    cur = conn.cursor()
    query = f"SELECT {input.agg_func}([{input.column}]) FROM {input.table}"
    cur.execute(query)
    val = cur.fetchone()[0]
    conn.close()
    return SummaryOutput(summary=val)


# Mount the MCP server in FastAPI
app.mount("/mcp", mcp.asgi_app())


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
