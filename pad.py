clean_json = llm_decision.strip()
if "```" in clean_json:
    clean_json = clean_json.replace("```json", "").replace("```", "")
parsed = json.loads(clean_json)


import json, ast, re

def safe_parse_llm_output(raw_text: str):
    """
    Safely parse LLM JSON-like output that may contain single quotes or markdown formatting.
    Supports both JSON and Python dict syntax.
    """
    if not raw_text:
        raise ValueError("Empty LLM output")

    # Clean code fences and whitespace
    clean = raw_text.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()

    # If multiple JSON/dict-like blocks, take the last one
    matches = re.findall(r"\{.*\}", clean, flags=re.DOTALL)
    if matches:
        clean = matches[-1]

    # Try strict JSON parsing
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Try Python literal evaluation (handles single quotes)
    try:
        return ast.literal_eval(clean)
    except Exception as e:
        raise ValueError(f"Unable to parse LLM output: {e}\nRaw text: {raw_text}")

parsed = safe_parse_llm_output(llm_decision)
