import streamlit as st
import asyncio
import json
import os
import traceback
from dotenv import load_dotenv
from fastmcp import Client
from openai import AsyncAzureOpenAI

load_dotenv()

# Azure OpenAI setup
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# MCP endpoints
MCP_OPS_FINOPS_URL = os.getenv("OPS_FINOPS_MCP_URL", "http://localhost:8500/mcp")
MCP_INFRA_URL = os.getenv("INFRA_MCP_URL", "http://localhost:9500/mcp")

# Initialize LLM client
llm_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-05-01-preview",
)


# -------------------------------------------------------------
# 1️. Utility: Ask LLM to decide intent
# -------------------------------------------------------------
async def ask_llm(user_input: str) -> dict:
    """Ask LLM to decide which workflow or tool to execute."""
    system_prompt = """
You are the Orchestrator for the Unified MCP Server system.

You can decide between:
1. Single-tool operations
2. The special 'update_rightsize_workflow'

If user asks something like "update the right-size variables in repository repo_123 for environment envX",
return this JSON:
{
  "workflow": "update_rightsize_workflow",
  "params": {
      "repo": "repo_123",
      "environment": "envX"
  }
}

If user just greets or asks general stuff:
{
  "workflow": "none",
  "message": "Hi there!  I can help you automate FinOps and Infra updates."
}

Otherwise return a normal single-tool call:
{
  "workflow": "single_tool",
  "tool": "tool_name",
  "params": {...}
}

Output strictly JSON — no markdown, no commentary.
"""

    response = await llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )

    output = response.choices[0].message.content.strip()
    return json.loads(output.replace("```json", "").replace("```", "").strip())


# -------------------------------------------------------------
# 2️. MCP helper
# -------------------------------------------------------------
async def call_mcp_tool(mcp_url: str, tool_name: str, params: dict):
    """Call any MCP server tool and return clean result."""
    async with Client(mcp_url) as client:
        response = await client.call_tool(tool_name, params)
        # Return simplified display form
        if isinstance(response, dict) and "columns" in response and "rows" in response:
            import pandas as pd
            df = pd.DataFrame(response["rows"], columns=response["columns"])
            return df.to_markdown(index=False)
        return response


# -------------------------------------------------------------
# 3️. Orchestration logic for workflows
# -------------------------------------------------------------
# async def run_update_rightsize_workflow(repo: str, environment: str):
#     """Full workflow: FinOps → InfraLogic → GitOps."""
#     log = []

#     # Step 1: Get FinOps right-sizing recommendations
#     log.append("🔹 Fetching FinOps recommendations...")
#     finops_data = await call_mcp_tool(
#         MCP_OPS_FINOPS_URL,
#         "get_rightsizing_recommendations",
#         {"environment": environment, "humanaID": repo},
#     )

#     log.append(" Got right-sizing recommendations.")

#     # Step 2: Clone repo (InfraLogic MCP)
#     log.append("🔹 Cloning repository...")
#     clone_result = await call_mcp_tool(
#         MCP_OPS_FINOPS_URL,  # GitOps tools are here
#         "clone_repo",
#         {"repo_url": f"https://github.com/org/{repo}.git"},
#     )
#     repo_path = clone_result.get("local_path") if isinstance(clone_result, dict) else clone_result
#     log.append(f" Repo cloned to {repo_path}")

#     # Step 3: Analyze Terraform
#     log.append(" Analyzing Terraform files...")
#     analysis_result = await call_mcp_tool(
#         MCP_INFRA_URL,
#         "analyze_infra",
#         {"repo_path": repo_path},
#     )
#     analysis_path = analysis_result.get("analysis_path") if isinstance(analysis_result, dict) else analysis_result
#     log.append(f" Analysis complete → {analysis_path}")

#     # Step 4: Update Terraform files
#     log.append(" Updating Terraform variables...")
#     update_result = await call_mcp_tool(
#         MCP_INFRA_URL,
#         "update_infra",
#         {
#             "repo_path": repo_path,
#             "analysis_path": analysis_path,
#             "recommendations": finops_data if isinstance(finops_data, dict) else {},
#         },
#     )
#     log.append(" Terraform files updated.")

#     # Step 5: Commit and push
#     log.append(" Committing changes...")
#     await call_mcp_tool(MCP_OPS_FINOPS_URL, "commit_and_push", {"commit_message": "Auto right-sizing update"})
#     log.append(" Changes committed and pushed.")

#     # Step 6: Create pull request
#     log.append(" Creating pull request...")
#     await call_mcp_tool(
#         MCP_OPS_FINOPS_URL,
#         "create_pull_request",
#         {"branch_name": "auto-rightsize", "title": "Auto Right-Sizing Update", "body": "Automated FinOps adjustments"},
#     )
#     log.append(" Pull request created successfully.")

#     return "\n".join(log)

async def run_update_rightsize_workflow(repo: str, environment: str, base_branch: str = "main"):
    """
    Orchestrated workflow with concurrency + error handling:
      1) FinOps recommendations (concurrent)
      2) Clone repo (concurrent)
      3) Analyze infra (serial)
      4) Update infra (serial)
      5) Commit + push + PR (only if update succeeded)
    If update fails: don't commit; optionally call cleanup_clone on GitOps MCP.
    """
    logs = []
    clone_result = None
    finops_data = None
    repo_path = None
    analysis_path = None

    logs.append(f"Workflow started for repo='{repo}' env='{environment}'")

    # --- Step A: Run FinOps lookup + Clone concurrently ---
    logs.append("Step A: Running FinOps lookup and cloning repo concurrently...")
    finops_task = asyncio.create_task(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "get_rightsizing_recommendations", {"environment": environment, "humana_id": repo})
    )
    # clone via GitHub/GitOps tools exposed by Ops & FinOps MCP (same as before)
    clone_task = asyncio.create_task(
        call_mcp_tool(MCP_OPS_FINOPS_URL, "clone_repo", {"repo_url": f"https://github.com/your-org/{repo}.git", "branch": base_branch})
    )

    # Wait concurrently and capture exceptions
    done, pending = await asyncio.wait([finops_task, clone_task], return_when=asyncio.ALL_COMPLETED)

    # Retrieve results and handle exceptions
    finops_exc = None
    clone_exc = None
    for t in done:
        if t is finops_task:
            try:
                finops_data = t.result()
                logs.append("FinOps recommendations fetched.")
            except Exception as e:
                finops_exc = e
                logs.append(f" FinOps lookup failed: {e}")
        elif t is clone_task:
            try:
                clone_result = t.result()
                # clone_result is expected to be dict with local_path or string
                if isinstance(clone_result, dict) and "local_path" in clone_result:
                    repo_path = clone_result["local_path"]
                else:
                    # attempt interpret string or nested data
                    if isinstance(clone_result, str):
                        repo_path = clone_result
                    elif isinstance(clone_result, dict) and "data" in clone_result:
                        repo_path = clone_result["data"].get("local_path") or clone_result["data"].get("path")
                logs.append(f"Repo cloned to: {repo_path}")
            except Exception as e:
                clone_exc = e
                logs.append(f" Clone failed: {e}")

    # If clone failed -> abort (nothing to rollback because nothing changed)
    if clone_exc:
        logs.append("Aborting workflow due to clone failure. No changes were made.")
        # include stack trace for debugging
        logs.append(traceback.format_exc())
        return "\n".join(logs)

    # If finops failed -> we can still proceed only if LLM/orch strategy permits, but safer to abort
    if finops_exc:
        logs.append("Aborting workflow due to FinOps lookup failure.")
        logs.append(traceback.format_exc())
        return "\n".join(logs)

    # --- Step B: Analyze infra ---
    logs.append("Step B: Analyzing Terraform files (InfraLogic MCP)...")
    try:
        analysis_result = await call_mcp_tool(MCP_INFRA_URL, "analyze_infra", {"repo_path": repo_path})
        # parse analysis_result for path
        if isinstance(analysis_result, dict) and "analysis_path" in analysis_result:
            analysis_path = analysis_result["analysis_path"]
        else:
            # fallback: if server returned raw string path
            analysis_path = analysis_result if isinstance(analysis_result, str) else None
        logs.append(f"Analysis completed. Analysis file: {analysis_path}")
    except Exception as e:
        logs.append(f" Analysis failed: {e}")
        logs.append("Aborting workflow. No changes will be committed.")
        # optional cleanup attempt
        try:
            await call_mcp_tool(MCP_OPS_FINOPS_URL, "cleanup_clone", {"local_path": repo_path})
            logs.append("Cleanup attempted: clone removed on MCP host.")
        except Exception:
            logs.append("Cleanup not available or failed (ignored).")
        logs.append(traceback.format_exc())
        return "\n".join(logs)

    # --- Step C: Update infra using FinOps recommendations ---
    logs.append("Step C: Updating Terraform files with FinOps recommendations (InfraLogic MCP)...")
    update_success = False
    update_result = None
    try:
        # Ensure recommendations is a dict (server expects dict)
        recs = finops_data if isinstance(finops_data, dict) else {}
        update_result = await call_mcp_tool(MCP_INFRA_URL, "update_infra", {
            "repo_path": repo_path,
            "analysis_path": analysis_path,
            "recommendations": recs
        })
        update_success = True
        logs.append(" Update completed successfully.")
        # Optionally log update_result summary
        logs.append(f"Update result: {update_result}")
    except Exception as e:
        logs.append(f" Update failed: {e}")
        logs.append("Will NOT commit or push any changes. Performing rollback/cleanup if possible.")
        # try best-effort cleanup on MCP host (optional tool)
        try:
            await call_mcp_tool(MCP_OPS_FINOPS_URL, "cleanup_clone", {"local_path": repo_path})
            logs.append("Cleanup attempted: clone removed on MCP host.")
        except Exception:
            logs.append("Cleanup tool not available or cleanup failed (ignored).")
        logs.append(traceback.format_exc())
        return "\n".join(logs)

    # --- Step D: Commit + Push + PR (only if update successful) ---
    if update_success:
        logs.append("Step D: Committing changes (Ops & FinOps MCP)...")
        try:
            # create feature branch first (safe practice)
            feature_branch = f"auto-rightsize-{environment}"
            await call_mcp_tool(MCP_OPS_FINOPS_URL, "create_branch", {"repo_path": repo_path, "branch_name": feature_branch, "base_branch": base_branch})
            logs.append(f"Feature branch '{feature_branch}' created.")

            # commit & push
            await call_mcp_tool(MCP_OPS_FINOPS_URL, "commit_and_push", {
                "repo_path": repo_path,
                "branch": feature_branch,
                "commit_message": "Automated rightsizing update"
            })
            logs.append(" Changes committed and pushed.")

            # create PR
            pr = await call_mcp_tool(MCP_OPS_FINOPS_URL, "create_pull_request", {
                "repo_fullname": repo,
                "source_branch": feature_branch,
                "target_branch": base_branch,
                "title": "Automated Right-Sizing Update",
                "body": "This PR updates Terraform variables based on FinOps recommendations."
            })
            logs.append(f" Pull request created: {pr.get('pr_url') if isinstance(pr, dict) else pr}")
        except Exception as e:
            logs.append(f" Commit/Push/PR failed: {e}")
            logs.append("IMPORTANT: The repo contains updated files locally on MCP host but push/PR failed.")
            # Do NOT attempt to push again automatically — leave for operator
            try:
                await call_mcp_tool(MCP_OPS_FINOPS_URL, "cleanup_clone", {"local_path": repo_path})
                logs.append("Cleanup attempted after failed push (best-effort).")
            except Exception:
                logs.append("Cleanup not available or failed (ignored).")
            logs.append(traceback.format_exc())
            return "\n".join(logs)

    # Success
    logs.append("Workflow completed successfully.")
    return "\n".join(logs)

# -------------------------------------------------------------
# 4️. Main orchestrator
# -------------------------------------------------------------
async def orchestrate(user_input: str):
    llm_output = await ask_llm(user_input)
    workflow = llm_output.get("workflow")

    if workflow == "none":
        return llm_output.get("message")

    elif workflow == "update_rightsize_workflow":
        repo = llm_output["params"]["repo"]
        env = llm_output["params"]["environment"]
        return await run_update_rightsize_workflow(repo, env)

    elif workflow == "single_tool":
        tool = llm_output["tool"]
        params = llm_output["params"]
        return await call_mcp_tool(MCP_OPS_FINOPS_URL, tool, params)

    else:
        return " Unknown request type."


# -------------------------------------------------------------
# 5️. Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="Unified MCP Orchestrator", layout="wide")
st.title("🤖 Unified MCP Orchestrator")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "activity" not in st.session_state:
    st.session_state.activity = []

# Sidebar logs
st.sidebar.header(" MCP Activity Log")
for entry in st.session_state.activity:
    st.sidebar.write(entry)

# Chat UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your command..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running workflow..."):
            result = asyncio.run(orchestrate(prompt))
            st.session_state.activity.append(result)
            st.write(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
