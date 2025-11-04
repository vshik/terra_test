import pytest
from unittest.mock import patch, MagicMock
import mcp_server as mcp_server_module

# 1. happy path returns columns and rows
@patch("mcp_server.pyodbc.connect")
def test_get_rightsizing_recommendations_success(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("VM","eastus","D2","D1",100.0)
    ]
    mock_cursor.description = [("resource_type",),("region",),("current_size",),("recommended_size",),("monthly_saving_usd",)]
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    params = mcp_server_module.FinOpsQueryInput(resource_type="VM", region="eastus", limit=1)
    res = mcp_server_module.get_rightsizing_recommendations.fn(params, None)
    assert isinstance(res, mcp_server_module.FinOpsQueryOutput)
    assert res.columns[0] == "resource_type"
    assert res.rows[0][0] == "VM"

# 2. empty result
@patch("mcp_server.pyodbc.connect")
def test_get_rightsizing_recommendations_empty(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.description = []
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    params = mcp_server_module.FinOpsQueryInput(resource_type="X", region="nowhere")
    res = mcp_server_module.get_rightsizing_recommendations.fn(params, None)
    assert res.rows == [] and res.columns == []

# 3. db connection error handled (raises)
@patch("mcp_server.pyodbc.connect", side_effect=Exception("conn failed"))
def test_get_rightsizing_recommendations_db_error(mock_connect):
    params = mcp_server_module.FinOpsQueryInput(resource_type="VM", region="eastus")
    with pytest.raises(Exception):
        mcp_server_module.get_rightsizing_recommendations.fn(params, None)
