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


async def ask_llm(user_input: str) -> str:
    """Ask LLM to decide which tool to call or return a message."""
    system_prompt = """
You are the Orchestrator for the Unified MCP Server that controls FinOps and GitHub tools.

Your job:
- Decide which MCP tool (if any) to invoke based on the user's request.
- If a tool applies, return its name and parameters.
- If not, return a human-friendly message instead.

Available tools:
1. get_rightsizing_recommendations(environment, resource_id)
2. validate_cost_savings(recommendation_id)
3. estimate_savings(resource_id)
4. clone_repo(repo_url)
5. create_pull_request(branch_name, title, body)
6. commit_and_push(commit_message)

Output format (JSON or Python dict):
- If a tool applies:
  {"tool": "tool_name", "params": {"param1": "value1"}}

- If no tool applies:
  {"tool": null, "message": "Your friendly message here"}

Examples:
User: "Get rightsizing for prod appsvc"
→ {"tool": "get_rightsizing_recommendations", "params": {"environment": "prod", "resource_id": "appsvc"}}

User: "hi"
→ {"tool": null, "message": "Hi there! I can help you run GitHub or FinOps operations such as fetching cost recommendations or creating pull requests."}

User: "what can you do"
→ {"tool": null, "message": "I can perform FinOps analytics and GitHub operations via the MCP tools."}

Always output exactly one JSON or Python dict block.
"""

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def safe_parse_llm_output(output: str):
    """Safely parse LLM output into a dict."""
    output = output.strip()
    try:
        # Handle JSON-like or Python dict-like output
        cleaned = output.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            # Try evaluating as Python dict (single quotes)
            return eval(cleaned)
        except Exception:
            return {"tool": None, "message": f"Could not parse LLM output: {output}"}
