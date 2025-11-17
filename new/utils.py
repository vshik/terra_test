# orchestrator/utils.py
import json, ast, re, asyncio, os
from typing import Any

def safe_parse_llm_output(raw_text: str):
    """Parse LLM output that is expected to contain a JSON/dict. Robust to fences and single quotes."""
    if not raw_text or not raw_text.strip():
        return {"tool": "none", "params": {}, "message": "Empty LLM response"}
    clean = raw_text.strip().replace("```json", "").replace("```", "").strip()
    # If there are multiple braces, take the last {...} block
    matches = re.findall(r"\{.*\}", clean, flags=re.DOTALL)
    if matches:
        clean = matches[-1]
    # Try strict JSON
    try:
        return json.loads(clean)
    except Exception:
        pass
    # Try Python literal
    try:
        return ast.literal_eval(clean)
    except Exception:
        # final fallback - return friendly none
        return {"tool": "none", "params": {}, "message": "Sorry, could not parse LLM response."}

def serialize_result(result: Any):
    """Make basic Python / Pydantic results JSON serializable for logging and UI."""
    try:
        from pydantic import BaseModel
    except Exception:
        BaseModel = None

    if result is None:
        return None
    if BaseModel and isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, (dict, list, str, int, float, bool)):
        return result
    # Attempt .dict() or .model_dump if available
    try:
        if hasattr(result, "dict"):
            return result.dict()
    except Exception:
        pass
    # fallback to string
    return str(result)
