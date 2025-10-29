import json
import os
from functools import lru_cache

QUERY_FILE = os.path.join(os.path.dirname(__file__), "../queries/sql_queries.json")

@lru_cache(maxsize=None)
def _load_all_queries(file_path: str = DEFAULT_QUERY_FILE) -> dict:
    """Load and cache all queries from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Query file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    return data


def load_query_from_json(query_name: str, file_path: str = QUERY_FILE) -> str:
    """Return a single query by name from JSON file."""
    queries = _load_all_queries(file_path)
    query = queries.get(query_name)
    if not query:
        raise KeyError(f"Query '{query_name}' not found in {file_path}")
    return query.strip()
