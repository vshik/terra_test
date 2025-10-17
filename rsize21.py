###############################################################################################################################
# rsize2.py:  regex python functions for updating locals{} type tf file + ruamel python function for updating yaml type tf file
###############################################################################################################################


import json
import re
from difflib import SequenceMatcher
import yaml
from ruamel.yaml import YAML
import json
import re
from copy import deepcopy
from ruamel.yaml.compat import StringIO


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """
    Update Terraform YAML variables for all instance_ids across environments.
    Preserves exact spacing/comments, updates only changed lines.
    Fully structure-agnostic (auto-detects top-level key).
    """

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # Prevent wrapping

    # --- Load YAML + preserve original lines ---
    with open(terraform_file, "r") as f:
        original_lines = f.readlines()
        yaml_data = yaml.load("".join(original_lines))
    print("Loaded YAML file.")

    # --- Load metadata JSON ---
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    print("Loaded metadata JSON.")

    known_envs = sorted(
        set(v.get("environment") for v in metadata.get("sizing_variables", []) if v.get("environment"))
    )
    print(f"Known environments (from metadata): {known_envs}")

    total_updates = 0
    env_updates = {}

    # --- Recursive Helper functions ---    
    def update_variable_recursive_all(data, target_key, new_val, path=None):
        """Recursively find and record updates, capturing YAML line numbers."""
        nonlocal total_updates
        if path is None:
            path = []
        results = []

        if isinstance(data, dict):
            for k, v in data.items():
                if k == target_key:
                    old_val = data[k]
                    data[k] = new_val
                    try:
                        line_no = data.lc.key(k)[0] + 1
                    except Exception:
                        line_no = None
                    results.append((line_no, old_val, new_val, path + [k]))
                    total_updates += 1
                if isinstance(v, (dict, list)):
                    results.extend(update_variable_recursive_all(v, target_key, new_val, path + [k]))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                results.extend(update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]))
        return results

    def find_all_blocks(data, path=None):
        """Find all dict-like sub-blocks."""
        if path is None:
            path = []
        blocks = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    blocks.append((k, v, path + [k]))
                    blocks.extend(find_all_blocks(v, path + [k]))
                elif isinstance(v, list):
                    for idx, item in enumerate(v):
                        blocks.extend(find_all_blocks(item, path + [f"{k}[{idx}]"]))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                blocks.extend(find_all_blocks(item, path + [f"[{idx}]"]))
        return blocks

    def detect_env_root(yaml_dict, env_name):
        """Automatically detect correct environment root — structure agnostic."""
        if not isinstance(yaml_dict, dict):
            return None
        # Case 1: environment is at top level
        if env_name in yaml_dict:
            return yaml_dict[env_name]
        # Case 2: environment nested under some top-level key
        for top_key, top_val in yaml_dict.items():
            if isinstance(top_val, dict) and env_name in top_val:
                return top_val[env_name]
        # Case 3: environment nested deeper
        for top_key, top_val in yaml_dict.items():
            if isinstance(top_val, dict):
                deeper = detect_env_root(top_val, env_name)
                if deeper is not None:
                    return deeper
        return None

    # --- Process all updates ---
    for update_item in updates:
        env_update = update_item["environment_name"]
        target_var = update_item["variable_name"]
        new_val = update_item["variable_value"]

        print(f"\n=== Processing update: {target_var} for environment '{env_update}' → {new_val} ===")

        # Collect instance_ids matching environment & variable
        instance_scores = {}
        for var in metadata.get("sizing_variables", []):
            if var.get("environment") == env_update:
                score = similarity(target_var, var.get("id", ""))
                if score >= similarity_threshold:
                    iid = var.get("instance_id")
                    if iid not in instance_scores or score > instance_scores[iid]:
                        instance_scores[iid] = score

        if not instance_scores:
            print(f"No metadata instance_ids found for {target_var} in environment '{env_update}'.")
            continue

        instance_ids = sorted(instance_scores.items(), key=lambda x: x[1], reverse=True)
        matched_ids = [iid for iid, _ in instance_ids]
        print(f"Matched metadata instance_ids: {matched_ids}")

        # --- Auto-detect environment dictionary ---
        env_dict = detect_env_root(yaml_data, env_update)
        if env_dict is None:
            print(f"Environment '{env_update}' not found in YAML.")
            continue
        print(f"Inside environment '{env_update}' in YAML structure.")

        # --- Find and update blocks ---
        all_blocks = find_all_blocks(env_dict)
        updates_to_apply = []
        seen_paths = set()

        for instance_id, _ in instance_ids:
            for block_name, block_dict, block_path in all_blocks:
                if instance_id and instance_id not in block_name:
                    continue
                changes = update_variable_recursive_all(block_dict, target_var, new_val, path=[env_update] + block_path)
                for line_no, old_val, val_new, path_list in changes:
                    if line_no and tuple(path_list) not in seen_paths:
                        updates_to_apply.append((line_no, old_val, val_new, path_list))
                        seen_paths.add(tuple(path_list))

        # --- Apply hybrid line patch (preserve all formatting) ---
        for (line_no, old_val, val_new, path_list) in updates_to_apply:
            if line_no is None or line_no - 1 >= len(original_lines):
                continue
            old_line = original_lines[line_no - 1]

            # Hardened regex for mixed indentation and inline comments
            pattern = re.compile(rf'^(\s*{re.escape(target_var)}\s*:\s*)([^\n#]*)(.*)$')
            new_line = pattern.sub(lambda m: f"{m.group(1)}{val_new}{m.group(3)}", old_line)

            if new_line != old_line:
                original_lines[line_no - 1] = new_line
                print(f"Updated {target_var}: {old_val} → {val_new} at path: {' → '.join(path_list)}, YAML line: {line_no}")

        if updates_to_apply:
            env_updates.setdefault(env_update, 0)
            env_updates[env_update] += len(updates_to_apply)

    # --- Write out patched file exactly as before ---
    with open(terraform_file, "w") as f:
        f.writelines(original_lines)

# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform YAML variables for all instance_ids across dev/qa/prod
#     using ruamel.yaml for line tracking and YAML preservation.
#     Ensures each variable_name is updated only once per instance_id.
#     Applies precise line patch so only updated lines are changed.
#     """

#     yaml = YAML()
#     yaml.preserve_quotes = True
#     yaml.width = 4096  # prevent line wrapping

#     # --- Load YAML file ---
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.load(f)
#     print("Loaded YAML file.")

#     # --- Read original YAML lines for precise patching ---
#     with open(terraform_file, "r") as f:
#         yaml_lines = f.readlines()

#     # --- Load metadata file ---
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print("Loaded metadata JSON.")

#     # --- Collect known environments ---
#     known_envs = sorted(
#         set(v.get("environment") for v in metadata.get("sizing_variables", []) if v.get("environment"))
#     )
#     print(f"Known environments (from metadata): {known_envs}")

#     # === Stats tracking ===
#     env_updates = {}
#     total_updates = 0

#     # --- Store which YAML line numbers need to be changed ---
#     changed_lines = {}

#     # === Recursive updater ===
#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         """Recursively update variable and track line numbers for precise patch."""
#         nonlocal total_updates
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     if old_val != new_val:
#                         data[k] = new_val
#                         try:
#                             line_no = data.lc.key(k)[0] + 1
#                         except Exception:
#                             line_no = None
#                         if line_no is not None:
#                             changed_lines[line_no] = (old_val, new_val)
#                         print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}, YAML line: {line_no if line_no else '?'}")
#                         total_updates += 1
#                         updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True

#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True

#         return updated

#     # === Recursive block finder ===
#     def find_all_blocks(data, path=None):
#         """Find all dict-like 'blocks' in YAML structure."""
#         if path is None:
#             path = []
#         blocks = []
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, dict):
#                     blocks.append((k, v, path + [k]))
#                     blocks.extend(find_all_blocks(v, path + [k]))
#                 elif isinstance(v, list):
#                     for idx, item in enumerate(v):
#                         blocks.extend(find_all_blocks(item, path + [f"{k}[{idx}]"]))
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 blocks.extend(find_all_blocks(item, path + [f"[{idx}]"]))
#         return blocks

#     # === Process each update item ===
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for environment '{env_update}' → {new_val} ===")

#         # --- Gather all unique instance_ids for the environment ---
#         instance_scores = {}
#         for var in metadata.get("sizing_variables", []):
#             if var.get("environment") == env_update:
#                 score = similarity(target_var, var.get("id", ""))
#                 if score >= similarity_threshold:
#                     iid = var.get("instance_id")
#                     # Keep only highest-scoring entry per instance_id
#                     if iid not in instance_scores or score > instance_scores[iid]:
#                         instance_scores[iid] = score

#         if not instance_scores:
#             print(f"No metadata instance_ids found for {target_var} in environment '{env_update}'.")
#             continue

#         # --- Deduplicated, sorted instance_ids ---
#         instance_ids = sorted(instance_scores.items(), key=lambda x: x[1], reverse=True)
#         matched_ids = [iid for iid, _ in instance_ids]
#         print(f"Matched metadata instance_ids: {matched_ids}")

#         # --- Locate environment node ---
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found in YAML.")
#             continue
#         print(f"Inside environment '{env_update}' in YAML structure.")

#         # --- Find all blocks in this environment ---
#         all_blocks = find_all_blocks(env_dict)

#         updated_this_env = 0

#         # --- Update all blocks that match instance_ids ---
#         for instance_id, score in instance_ids:
#             matched = False
#             for block_name, block_dict, block_path in all_blocks:
#                 if instance_id and instance_id not in block_name:
#                     continue
#                 if update_variable_recursive_all(block_dict, target_var, new_val, path=[env_update] + block_path):
#                     matched = True
#                     updated_this_env += 1
#                     break  # prevent re-updating same block
#             if not matched:
#                 print(f"Variable '{target_var}' not found under instance_id '{instance_id}' in '{env_update}'.")

#         if updated_this_env > 0:
#             env_updates.setdefault(env_update, 0)
#             env_updates[env_update] += updated_this_env

#     # --- Apply precise line patch ---
#     patched_lines = []
#     for idx, line in enumerate(yaml_lines, start=1):
#         if idx in changed_lines:
#             old, new = changed_lines[idx]
#             # Regex: preserve everything before/after value including comments
#             pattern = rf"(^[ \t]*[^#:\n]+:\s*){re.escape(str(old))}(\s*(?:#.*)?$)"
#             updated_line = re.sub(pattern, rf"\g<1>{new}\g<2>", line)   # use g<1> and g<2> to avoid numeric group errors in regex
#             patched_lines.append(updated_line)
#         else:
#             patched_lines.append(line)

#     # --- Write patched YAML lines back ---
#     with open(terraform_file, "w") as f:
#         f.writelines(patched_lines)


# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform YAML variables for all instance_ids across dev/qa/prod
#     using ruamel.yaml for line tracking and YAML preservation.
#     Ensures each variable_name is updated only once per instance_id.
#     """

#     yaml = YAML()
#     yaml.preserve_quotes = True
#     yaml.width = 4096  # Prevent long strings from breaking to next line
#     yaml.explicit_start = False
#     yaml.indent(mapping=2, sequence=4, offset=2)
#     yaml.scalarstring_preserve_quotes = True
#     yaml.allow_unicode = True
#     yaml.default_flow_style = False
#     yaml.default_style = None
#     yaml.representer.ignore_aliases = lambda *args: True

#     # --- Load YAML file ---
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.load(f)
#     print("Loaded YAML file.")

#     # --- Load metadata file ---
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print("Loaded metadata JSON.")

#     # --- Collect known environments ---
#     known_envs = sorted(
#         set(v.get("environment") for v in metadata.get("sizing_variables", []) if v.get("environment"))
#     )
#     print(f"Known environments (from metadata): {known_envs}")

#     # === Stats tracking ===
#     env_updates = {}
#     total_updates = 0

#     # === Recursive updater ===
#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         """Recursively update variable and print accurate YAML line numbers."""
#         nonlocal total_updates
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     try:
#                         line_no = data.lc.key(k)[0] + 1
#                     except Exception:
#                         line_no = '?'
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}, YAML line: {line_no}")
#                     total_updates += 1
#                     updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True

#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True

#         return updated

#     # === Recursive block finder ===
#     def find_all_blocks(data, path=None):
#         """Find all dict-like 'blocks' in YAML structure."""
#         if path is None:
#             path = []
#         blocks = []
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, dict):
#                     blocks.append((k, v, path + [k]))
#                     blocks.extend(find_all_blocks(v, path + [k]))
#                 elif isinstance(v, list):
#                     for idx, item in enumerate(v):
#                         blocks.extend(find_all_blocks(item, path + [f"{k}[{idx}]"]))
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 blocks.extend(find_all_blocks(item, path + [f"[{idx}]"]))
#         return blocks

#     # === Process each update item ===
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for environment '{env_update}' → {new_val} ===")

#         # --- Gather all unique instance_ids for the environment ---
#         instance_scores = {}
#         for var in metadata.get("sizing_variables", []):
#             if var.get("environment") == env_update:
#                 score = similarity(target_var, var.get("id", ""))
#                 if score >= similarity_threshold:
#                     iid = var.get("instance_id")
#                     # Keep only highest-scoring entry per instance_id
#                     if iid not in instance_scores or score > instance_scores[iid]:
#                         instance_scores[iid] = score

#         if not instance_scores:
#             print(f"No metadata instance_ids found for {target_var} in environment '{env_update}'.")
#             continue

#         # --- Deduplicated, sorted instance_ids ---
#         instance_ids = sorted(instance_scores.items(), key=lambda x: x[1], reverse=True)
#         matched_ids = [iid for iid, _ in instance_ids]
#         print(f"Matched metadata instance_ids: {matched_ids}")

#         # --- Locate environment node ---
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found in YAML.")
#             continue
#         print(f"Inside environment '{env_update}' in YAML structure.")

#         # --- Find all blocks in this environment ---
#         all_blocks = find_all_blocks(env_dict)

#         updated_this_env = 0

#         # --- Update all blocks that match instance_ids ---
#         for instance_id, score in instance_ids:
#             matched = False
#             for block_name, block_dict, block_path in all_blocks:
#                 if instance_id and instance_id not in block_name:
#                     continue
#                 if update_variable_recursive_all(block_dict, target_var, new_val, path=[env_update] + block_path):
#                     matched = True
#                     updated_this_env += 1
#                     break  # prevent re-updating same block
#             if not matched:
#                 print(f"Variable '{target_var}' not found under instance_id '{instance_id}' in '{env_update}'.")

#         if updated_this_env > 0:
#             env_updates.setdefault(env_update, 0)
#             env_updates[env_update] += updated_this_env

#     # --- Write YAML file back ---
#     with open(terraform_file, "w") as f:
#         yaml.dump(yaml_data, f)


# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform/YAML file variables based on metadata and update definitions,
#     preserving YAML line numbers using ruamel.yaml.
#     Updates all instance_id matches for a variable under the same environment.
#     """
#     yaml = YAML()
#     yaml.preserve_quotes = True

#     # --- Load YAML ---
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.load(f)
#     print(f"Loaded YAML file.")

#     # --- Load metadata ---
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print(f"Loaded metadata JSON.")

#     # --- Gather known environments ---
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])
#     print(f"Known environments (from metadata): {sorted(known_envs)}")

#     # Recursive functions follow...
#     # --- Recursive function to update variables ---
#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     line_no = getattr(data.lc, 'data', {}).get(k, [None, None])[0]
#                     line_no = line_no + 1 if line_no is not None else '?'
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}, YAML line: {line_no}")
#                     updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True
#         return updated

#     # --- Recursive function to find all blocks under an environment ---
#     def find_all_blocks(data, path=None):
#         if path is None:
#             path = []
#         blocks = []
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, dict):
#                     blocks.append((k, v))
#                     blocks.extend(find_all_blocks(v, path + [k]))
#                 elif isinstance(v, list):
#                     for idx, item in enumerate(v):
#                         blocks.extend(find_all_blocks(item, path + [f"{k}[{idx}]"]))
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 blocks.extend(find_all_blocks(item, path + [f"[{idx}]"]))
#         return blocks

#     # --- Process each update ---
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for environment '{env_update}' → {new_val} ===")

#         # --- Find best metadata match based on similarity ---
#         best_match = None
#         best_score = -1
#         for var in metadata.get("sizing_variables", []):
#             if var.get("environment") == env_update:
#                 score = similarity(target_var, var.get("id", ""))
#                 if score > best_score:
#                     best_score = score
#                     best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No metadata match for {target_var} in environment '{env_update}' (best_score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         instance_id = best_match.get("instance_id")
#         print(f"Matched metadata: lhs_var={lhs_var}, instance_id={instance_id}, best_score={best_score:.2f}")

#         # --- Locate environment ---
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found in YAML.")
#             continue
#         print(f"Inside environment '{env_update}' in YAML structure.")

#         # --- Update all blocks in environment matching instance_id ---
#         all_blocks = find_all_blocks(env_dict)
#         any_updated = False
#         for block_name, block_dict in all_blocks:
#             # Update if block matches instance_id or contains the variable
#             if instance_id is None or instance_id in block_name or lhs_var in block_dict:
#                 updated = update_variable_recursive_all(block_dict, lhs_var, new_val, path=[env_update, block_name])
#                 if updated:
#                     any_updated = True

#         if not any_updated:
#             print(f"Variable '{target_var}' not found under any block in environment '{env_update}'.")

#     # --- Write updated YAML back to file ---
#     with open(terraform_file, "w") as f:
#         yaml.dump(yaml_data, f)

#     print("\nYAML file updated successfully, all matching instance_ids updated.")

def update_terraform_file_locals_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
    """
    Update Terraform locals file values (nested HCL-style) using metadata-driven updates.

    This function:
      - Reads a Terraform locals file line-by-line.
      - Finds the correct environment, block, and sub-block context.
      - Updates only the intended variable inside that context.
      - Skips commented lines and preserves formatting.
      - Handles brace depth tracking automatically (even with top-level 'locals { ... }').

    Args:
        terraform_file: Path to the Terraform locals file to modify.
        metadata_file:  Path to JSON metadata containing sizing variables.
        updates:        List of update dictionaries with:
                            environment_name, resource_group_name, resource_name,
                            variable_name, variable_value
        similarity_threshold: Minimum fuzzy match score to consider variable names similar.

    Example usage:
        updates = [
            {
                "environment_name": "dev",
                "resource_group_name": "rg-dev-eastus2",
                "resource_name": "app-server",
                "variable_name": "vm-size",
                "variable_value": "Standard_D2s_v3"
            }
        ]
    """

    # --- Step 1: Load Terraform file and metadata ---
    with open(terraform_file, "r") as f:
        lines = f.readlines()
    updated_lines = lines[:]  # make a copy for modification

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Collect known environments from metadata (e.g., dev, qa, prd)
    known_envs = {v["environment"] for v in metadata.get("sizing_variables", []) if v.get("environment")}

    # Template for matching assignment lines like: usr-vm-size = "Standard_D2s_v3"
    assign_pattern_template = r"^(\s*{}\s*=\s*)(.*)"

    # --- Step 2: Process each update entry ---
    for update_item in updates:
        env_update = update_item["environment_name"]
        rg_update = update_item["resource_group_name"]
        rn_update = update_item["resource_name"]
        target_var = update_item["variable_name"]
        new_val = update_item["variable_value"]

        print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

        # --- Step 3: Match both environment and variable ID in metadata ---
        sizing_vars = metadata.get("sizing_variables", [])

        # Filter metadata entries for this environment (exact match)
        same_env_vars = [v for v in sizing_vars if v.get("environment") == env_update]

        # Prefer same environment; fallback to all if none found
        search_space = same_env_vars if same_env_vars else sizing_vars

        # Find best variable match by fuzzy similarity on the 'id' field
        best_match = None
        best_score = -1
        for var in search_space:
            score = similarity(target_var.lower(), var.get("id", "").lower())
            if score > best_score:
                best_score = score
                best_match = var

        # Skip if no good match found
        if not best_match or best_score < similarity_threshold:
            print(f"No good metadata match for {target_var} in env={env_update} (best score={best_score:.2f})")
            continue

        # --- Step 4: Extract match details ---
        lhs_var = best_match["id"]                   # e.g., usr-vm-size
        block_id = best_match.get("service")         # e.g., windowsvm
        sub_block_id = best_match.get("instance_id") # e.g., ent-member
        pattern = re.compile(assign_pattern_template.format(re.escape(lhs_var)))

        replaced = False
        brace_depth = 0               # global brace tracking
        locals_depth_offset = None    # offset for top-level 'locals {' block
        inside_env = inside_block = inside_subblock = False
        current_env = current_block = current_subblock = None

        # --- Step 5: Line-by-line parse and update ---
        for i, line in enumerate(updated_lines):
            # Skip commented lines
            if re.match(r'^\s*#', line) or re.match(r'^\s*/\*', line):
                continue

            # Detect start of top-level "locals" block
            if locals_depth_offset is None and re.search(r'^\s*locals\s*=\s*{|^\s*locals\s*{', line):
                locals_depth_offset = brace_depth + 1
                print(f"DEBUG: Found top-level locals at line {i+1}, setting offset={locals_depth_offset}")

            # Update brace depth counter
            open_braces = line.count("{")
            close_braces = line.count("}")
            brace_depth += open_braces - close_braces

            # Adjust for locals depth offset (so env depth == 1)
            effective_depth = brace_depth - (locals_depth_offset or 0)

            # --- Detect environment section (e.g., dev = { ) ---
            env_match = re.match(r'^\s*(\w+)\s*=\s*{', line)
            if env_match and effective_depth == 1:
                env_name = env_match.group(1)
                if env_name in known_envs:
                    current_env = env_name
                    inside_env = (env_name == env_update)
                    print(f"DEBUG: Entering env {current_env} at line {i+1}")

            # --- Detect block/service section (e.g., windowsvm = { ) ---
            block_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
            if block_match and inside_env and effective_depth == 2:
                blk_name = block_match.group(1)
                current_block = blk_name
                inside_block = (blk_name == block_id)
                print(f"DEBUG: Inside block {current_block} at line {i+1} (inside_block={inside_block})")

            # --- Detect sub-block/instance section (e.g., ent-member = { ) ---
            subblock_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
            if subblock_match and inside_block and effective_depth == 3:
                subblk = subblock_match.group(1)
                current_subblock = subblk
                inside_subblock = (subblk == sub_block_id)
                print(f"DEBUG: Inside sub-block {current_subblock} at line {i+1} (inside_subblock={inside_subblock})")

            # --- Apply variable update only inside correct env/block/sub-block ---
            if inside_env and inside_block and (sub_block_id == "default" or inside_subblock):
                m = pattern.match(line)
                if m:
                    lhs = m.group(1)
                    rhs = m.group(2).strip()

                    # Preserve quote style from original value
                    if rhs.startswith('"') and rhs.endswith('"'):
                        new_rhs = f'"{new_val}"'
                    elif rhs.startswith("'") and rhs.endswith("'"):
                        new_rhs = f"'{new_val}'"
                    else:
                        new_rhs = str(new_val)

                    # Replace line with new value
                    updated_lines[i] = f"{lhs}{new_rhs}\n"
                    replaced = True
                    print(f" Updated {lhs_var} in {env_update}/{block_id}/{sub_block_id} → {new_rhs} (line {i+1})")
                    break

        # --- Step 6: Post-check if update failed ---
        if not replaced:
            print(f" Could not update {lhs_var}: no matching env/block/subblock found")

    # --- Step 7: Write final output back to file ---
    with open(terraform_file, "w") as f:
        f.writelines(updated_lines)

def update_terraform_file_tfvars_type(terraform_file: str, updates: list, similarity_threshold: float = 0.55):    
    """
    Updates right-sizing variables in a Terraform .auto.tfvars file based on entries from updates list, using fuzzy matching.
    """
    with open(terraform_file, "r") as f:
        lines = f.readlines()

    updated_lines = lines[:]
    changes_made = []

    for update in updates:
        variable_name = update["variable_name"]
        variable_value = update["variable_value"]
        resource_group = update["resource_group_name"]
        resource_name = update["resource_name"]

        best_match = None
        best_score = 0
        match_index = None

        # Find best-matching variable in file
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.match(r'^([\w\-\_]+)\s*=\s*(.+)', stripped)
            if not match:
                continue

            var_in_file = match.group(1)
            score = similarity(var_in_file.lower(), variable_name.lower())

            if score > best_score:
                best_score = score
                best_match = var_in_file
                match_index = i

        # Skip if no close enough match
        if not best_match or best_score < similarity_threshold:
            print(f"Could not find a good match for '{variable_name}' (best_score={best_score:.2f})")
            continue

        old_line = lines[match_index]
        old_value = re.search(r'=\s*(.+)', old_line).group(1).strip()

        # Preserve type style (quotes, bool, null)
        if old_value.startswith('"') and old_value.endswith('"'):
            new_value_str = f'"{variable_value}"'
        elif old_value in ["true", "false", "null"] or re.match(r'^\d+$', old_value):
            new_value_str = str(variable_value)
        else:
            new_value_str = f'"{variable_value}"'

        new_line = re.sub(r'=\s*.+', f"= {new_value_str}", old_line)
        updated_lines[match_index] = new_line

        line_number = match_index + 1  # track line number
        changes_made.append((best_match, old_value, new_value_str, line_number, best_score))
        print(f"Updated {best_match}: {old_value} → {new_value_str} (score={best_score:.2f}) at Line {line_number}")

    # Write back to file
    with open(terraform_file, "w") as f:
        f.writelines(updated_lines)
    print(f"\n{len(changes_made)} variables updated successfully.")


# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform/YAML file variables based on metadata and update definitions.
#     """

#     # Load YAML file
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.safe_load(f)

#     # Load metadata JSON
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)

#     # Gather all known environments from metadata
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])
#     print(f"Known environments (from metadata): {sorted(known_envs)}")

#     # Recursive helper functions - to locate environment, to locate block and to update target variables
#     def find_environment_node(data, target_env, path=None):
#         """Recursively search for the dict node containing the target environment."""
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_env in data:
#                 # print(f"Found environment '{target_env}' under path: {' → '.join(path) or '[root]'}")
#                 return data[target_env]
#             for k, v in data.items():
#                 found = find_environment_node(v, target_env, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_environment_node(item, target_env, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def find_block_node(data, target_block, path=None):
#         """Recursively search for the dict node containing the target block/service."""
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_block in data:
#                 # print(f"Found block '{target_block}' under path: {' → '.join(path) or '[env root]'}")
#                 return data[target_block]
#             for k, v in data.items():
#                 found = find_block_node(v, target_block, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_block_node(item, target_block, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         """Recursively update all occurrences of target_key in nested dict/list."""
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}")
#                     updated = True
#                 # Recurse into nested structures
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True
#         return updated

#     # Parse and get all items in updates list
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

#         # Compare updates list with metadata and find best matching metadata variable (similarity-based)        
#         best_match = None
#         best_score = -1
#         for var in metadata.get("sizing_variables", []):
#             env_meta = var.get("environment", "")
#             var_id = var.get("id", "")
#             score = similarity(target_var, var_id)
#             if env_meta == env_update and score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env '{env_update}' (best_score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")
#         env_meta = best_match.get("environment")

#         print(f"Matched metadata: lhs_var={lhs_var}, block_id={block_id}, sub_block_id={sub_block_id}, env_meta={env_meta}")

#         # Locate environment node, first try under 'dpe', then general recursive search
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             env_dict = find_environment_node(yaml_data, env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found anywhere in YAML.")
#             continue
#         print(f"Inside environment '{env_update}' in YAML structure.")

#         # Locate block/service
#         block_dict = find_block_node(env_dict, block_id)
#         if block_dict is None:
#             print(f"Block/service '{block_id}' not found anywhere under environment '{env_update}'.")
#             continue
#         print(f"Inside block '{block_id}' in environment '{env_update}'.")

#         # Apply variable update recursively
#         if not update_variable_recursive_all(block_dict, lhs_var, new_val):
#             print(f"Variable '{lhs_var}' not found anywhere under block '{block_id}'. Skipping.")

#     # Write updated YAML back to file
#     with open(terraform_file, "w") as f:
#         yaml.safe_dump(yaml_data, f, sort_keys=False, default_flow_style=False)

#     print("\nYAML file updated successfully.")

# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform/YAML file variables based on metadata and update definitions.
#     """

#     # Load YAML file
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.safe_load(f)
#     print(f"Loaded YAML file. [line {traceback.extract_stack()[-1].lineno}]")

#     # Load metadata JSON
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print(f"Loaded metadata JSON. [line {traceback.extract_stack()[-1].lineno}]")

#     # Gather all known environments from metadata
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])
#     print(f"Known environments (from metadata): {sorted(known_envs)} [line {traceback.extract_stack()[-1].lineno}]")

#     # Recursive helper functions
#     def find_environment_node(data, target_env, path=None):
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_env in data:
#                 print(f"Found environment '{target_env}' under path: {' → '.join(path) or '[root]'} [line {traceback.extract_stack()[-1].lineno}]")
#                 return data[target_env]
#             for k, v in data.items():
#                 found = find_environment_node(v, target_env, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_environment_node(item, target_env, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def find_block_node(data, target_block, path=None):
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_block in data:
#                 print(f"Found block '{target_block}' under path: {' → '.join(path) or '[env root]'} [line {traceback.extract_stack()[-1].lineno}]")
#                 return data[target_block]
#             for k, v in data.items():
#                 found = find_block_node(v, target_block, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_block_node(item, target_block, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])} [line {traceback.extract_stack()[-1].lineno}]")
#                     updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True
#         return updated

#     # Process all updates
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} === [line {traceback.extract_stack()[-1].lineno}]")

#         # Match metadata
#         best_match = None
#         best_score = -1
#         for var in metadata.get("sizing_variables", []):
#             env_meta = var.get("environment", "")
#             var_id = var.get("id", "")
#             score = similarity(target_var, var_id)
#             if env_meta == env_update and score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env '{env_update}' (best_score={best_score:.2f}) [line {traceback.extract_stack()[-1].lineno}]")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")
#         env_meta = best_match.get("environment")

#         print(f"Matched metadata: lhs_var={lhs_var}, block_id={block_id}, sub_block_id={sub_block_id}, env_meta={env_meta} [line {traceback.extract_stack()[-1].lineno}]")

#         # Locate environment node
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             env_dict = find_environment_node(yaml_data, env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found anywhere in YAML. [line {traceback.extract_stack()[-1].lineno}]")
#             continue
#         print(f"Inside environment '{env_update}' in YAML structure. [line {traceback.extract_stack()[-1].lineno}]")

#         # Locate block/service
#         block_dict = find_block_node(env_dict, block_id)
#         if block_dict is None:
#             print(f"Block/service '{block_id}' not found anywhere under environment '{env_update}'. [line {traceback.extract_stack()[-1].lineno}]")
#             continue
#         print(f"Inside block '{block_id}' in environment '{env_update}'. [line {traceback.extract_stack()[-1].lineno}]")

#         # Update variable
#         if not update_variable_recursive_all(block_dict, lhs_var, new_val):
#             print(f"Variable '{lhs_var}' not found anywhere under block '{block_id}'. Skipping. [line {traceback.extract_stack()[-1].lineno}]")

#     # Save file
#     with open(terraform_file, "w") as f:
#         yaml.safe_dump(yaml_data, f, sort_keys=False, default_flow_style=False)
#     print(f"YAML file updated successfully. [line {traceback.extract_stack()[-1].lineno}]")

# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform/YAML file variables based on metadata and update definitions,
#     preserving YAML line numbers using ruamel.yaml.
#     """

#     yaml = YAML()
#     yaml.preserve_quotes = True  # preserve quotes

#     # Load YAML file
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.load(f)
#     print(f"Loaded YAML file.")

#     # Load metadata JSON
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print(f"Loaded metadata JSON.")

#     # Gather all known environments from metadata
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])
#     print(f"Known environments (from metadata): {sorted(known_envs)}")

#     # Recursive helper functions
#     def find_environment_node(data, target_env, path=None):
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_env in data:
#                 line_no = getattr(data[target_env].lc, 'line', '?') + 1
#                 print(f"Found environment '{target_env}' at YAML line {line_no} under path: {' → '.join(path) or '[root]'}")
#                 return data[target_env]
#             for k, v in data.items():
#                 found = find_environment_node(v, target_env, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_environment_node(item, target_env, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def find_block_node(data, target_block, path=None):
#         if path is None:
#             path = []

#         if isinstance(data, dict):
#             if target_block in data:
#                 line_no = getattr(data[target_block].lc, 'line', '?') + 1
#                 print(f"Found block '{target_block}' at YAML line {line_no} under path: {' → '.join(path) or '[env root]'}")
#                 return data[target_block]
#             for k, v in data.items():
#                 found = find_block_node(v, target_block, path + [k])
#                 if found is not None:
#                     return found
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 found = find_block_node(item, target_block, path + [f"[{idx}]"])
#                 if found is not None:
#                     return found
#         return None

#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     line_no = getattr(data.lc, 'data', {}).get(k, [None, None])[0]
#                     line_no = line_no + 1 if line_no is not None else '?'
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}, YAML line: {line_no}")
#                     updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True
#         return updated

#     # Process all updates
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

#         # Match metadata
#         best_match = None
#         best_score = -1
#         for var in metadata.get("sizing_variables", []):
#             env_meta = var.get("environment", "")
#             var_id = var.get("id", "")
#             score = similarity(target_var, var_id)
#             if env_meta == env_update and score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env '{env_update}' (best_score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")
#         env_meta = best_match.get("environment")

#         print(f"Matched metadata: lhs_var={lhs_var}, block_id={block_id}, sub_block_id={sub_block_id}, env_meta={env_meta}")

#         # Locate environment node
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             env_dict = find_environment_node(yaml_data, env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found anywhere in YAML.")
#             continue

#         # Locate block/service
#         block_dict = find_block_node(env_dict, block_id)
#         if block_dict is None:
#             print(f"Block/service '{block_id}' not found under environment '{env_update}'.")
#             continue

#         # Update variable
#         if not update_variable_recursive_all(block_dict, lhs_var, new_val):
#             print(f"Variable '{lhs_var}' not found under block '{block_id}'. Skipping.")

#     # Write updated YAML back
#     with open(terraform_file, "w") as f:
#         yaml.dump(yaml_data, f)

#     print("\nYAML file updated successfully with correct line numbers.")

# def update_terraform_file_yaml_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     """
#     Update Terraform/YAML file variables based on metadata and update definitions,
#     preserving YAML line numbers using ruamel.yaml.
#     Supports multiple instance_id matches for a given environment_name.
#     """

#     yaml = YAML()
#     yaml.preserve_quotes = True

#     # Load YAML
#     with open(terraform_file, "r") as f:
#         yaml_data = yaml.load(f)
#     print(f"Loaded YAML file.")

#     # Load metadata
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)
#     print(f"Loaded metadata JSON.")

#     # Gather known environments
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])
#     print(f"Known environments (from metadata): {sorted(known_envs)}")

#     # Recursive update function
#     def update_variable_recursive_all(data, target_key, new_val, path=None):
#         if path is None:
#             path = []
#         updated = False

#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k == target_key:
#                     old_val = data[k]
#                     data[k] = new_val
#                     line_no = getattr(data.lc, 'data', {}).get(k, [None, None])[0]
#                     line_no = line_no + 1 if line_no is not None else '?'
#                     print(f"Updated {target_key}: {old_val} → {new_val} at path: {' → '.join(path + [k])}, YAML line: {line_no}")
#                     updated = True
#                 if isinstance(v, (dict, list)):
#                     if update_variable_recursive_all(v, target_key, new_val, path + [k]):
#                         updated = True
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 if update_variable_recursive_all(item, target_key, new_val, path + [f"[{idx}]"]):
#                     updated = True
#         return updated

#     # Helper to find all blocks under an environment
#     def find_all_blocks(data, path=None):
#         if path is None:
#             path = []
#         blocks = []
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, dict):
#                     blocks.append((k, v))
#                     blocks.extend(find_all_blocks(v, path + [k]))
#                 elif isinstance(v, list):
#                     for idx, item in enumerate(v):
#                         blocks.extend(find_all_blocks(item, path + [f"{k}[{idx}]"]))
#         elif isinstance(data, list):
#             for idx, item in enumerate(data):
#                 blocks.extend(find_all_blocks(item, path + [f"[{idx}]"]))
#         return blocks

#     # Process all updates
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for environment '{env_update}' → {new_val} ===")

#         # Match metadata variable
#         matched_vars = []
#         for var in metadata.get("sizing_variables", []):
#             if var.get("environment") == env_update and similarity(target_var, var.get("id", "")) >= similarity_threshold:
#                 matched_vars.append(var)

#         if not matched_vars:
#             print(f"No metadata match for {target_var} in environment '{env_update}'")
#             continue

#         # Locate environment
#         env_dict = yaml_data.get("dpe", {}).get(env_update)
#         if env_dict is None:
#             print(f"Environment '{env_update}' not found in YAML.")
#             continue

#         # Iterate all blocks in environment
#         all_blocks = find_all_blocks(env_dict)
#         any_updated = False
#         for lhs_var_meta in matched_vars:
#             lhs_var = lhs_var_meta["id"]
#             instance_id = lhs_var_meta.get("instance_id")

#             for block_name, block_dict in all_blocks:
#                 # Update only if block matches instance_id or contains variable
#                 if instance_id is None or instance_id in block_name or lhs_var in block_dict:
#                     updated = update_variable_recursive_all(block_dict, lhs_var, new_val, path=[env_update, block_name])
#                     if updated:
#                         any_updated = True

#         if not any_updated:
#             print(f"Variable '{target_var}' not found under any block in environment '{env_update}'.")

#     # Write YAML back
#     with open(terraform_file, "w") as f:
#         yaml.dump(yaml_data, f)

#     print("\nYAML file updated successfully, all matching instance_ids updated.")

# def update_terraform_file_locals_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     # Load locals file
#     with open(terraform_file, "r") as f:
#         lines = f.readlines()
#     updated_lines = lines[:]

#     # Load metadata JSON    
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)

#     # Gather all known environments
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])

#     # Regex patterns to identify environments, blocks, and assignments
#     # env_pattern = re.compile(r'^\s*(\w+)\s*=\s*{')
#     # block_pattern = re.compile(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{')
#     # assign_pattern_template = r"^(\s*{}\s*=\s*)(.*)"
#     env_pattern = re.compile(r'^\s*(\w+)\s*=\s*{')                  # environment (e.g., npe = {)
#     block_pattern = re.compile(r'^\s{4}([A-Za-z0-9_\-]+)\s*=\s*{')  # block (exactly 4 spaces indent)
#     sub_block_pattern = re.compile(r'^\s{6,}([A-Za-z0-9_\-]+)\s*=\s*{')  # sub-block (≥6 spaces)
#     assign_pattern_template = r"^(\s*{}\s*=\s*)(.*)"

#     # Parse and get all items in updates list
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

#         # Compare updates list with metadata and find best matching metadata variable (similarity-based)
#         best_match = None
#         best_score = -1
#         sizing_vars = metadata.get("sizing_variables", [])
#         same_env_vars = [v for v in sizing_vars if v.get("environment") == env_update]    # filter for given environment in updates list
#         # If no same-env vars (i.e no vars found for given environment), fall back and search across all environments
#         search_space = same_env_vars if same_env_vars else sizing_vars
#         if not same_env_vars:
#             print(f"WARNING: No metadata entries found for environment '{env_update}', falling back to secrah across all environments.")

#         # Choose best variable name match within environment
#         for var in search_space:
#             score = similarity(target_var.lower(), var.get("id", "").lower())
#             if score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env={env_update} (best score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")
#         env_meta = best_match.get("environment")

#         pattern = re.compile(assign_pattern_template.format(re.escape(lhs_var))) # exact match with variable name

#         current_env = None
#         current_block = None
#         current_subblock = None
#         brace_depth = 0
#         replaced = False

#         inside_env = False
#         inside_block = False
#         inside_subblock = False

#         # iterate through lines to find correct context and update variable
#         for i, line in enumerate(updated_lines):
#             open_braces = line.count("{")
#             close_braces = line.count("}")

#             # Section for matching Environment 
#             env_match = env_pattern.match(line)
#             if env_match:
#                 env_name = env_match.group(1)
#                 if env_name in known_envs:
#                     current_env = env_name
#                     inside_env = (env_name == env_update)
#                     print(f"DEBUG: Entering env {current_env} at line {i+1}")

#             # Section for matching Block (resource/service)
#             block_match = block_pattern.match(line)
#             if block_match and inside_env:
#                 blk_name = block_match.group(1)
#                 current_block = blk_name
#                 inside_block = (blk_name == block_id)
#                 # print(f"DEBUG: Inside block {current_block} at line {i+1}, inside_block = {inside_block}, blk_name = {blk_name}, block_id = {block_id}")
#                 print(f"DEBUG: Inside block {current_block} at line {i+1}")

#             # Section for matching Sub-block (instance)
#             if inside_block and inside_env:
#                 # subblock_match = block_pattern.match(line)
#                 subblock_match = sub_block_pattern.match(line)
#                 if subblock_match:
#                     subblk = subblock_match.group(1)
#                     current_subblock = subblk
#                     inside_subblock = (subblk == sub_block_id)
#                     # print(f"DEBUG: Subblock {current_subblock} at line {i+1}, inside_subblock = {inside_subblock}")
#                     print(f"DEBUG: Inside Sub-block {current_subblock} at line {i+1}")

#             # Update brace depth and exit contexts if needed            
#             brace_depth += open_braces - close_braces
#             if brace_depth == 0:
#                 current_env = current_block = current_subblock = None
#                 inside_env = inside_block = inside_subblock = False

#             # Skip commented lines
#             if re.match(r'^\s*#', line) or re.match(r'^\s*/\*', line):
#                 continue

#             # Apply update only when inside correct context
#             # print(f"inside_env = {inside_env}  inside_block = {inside_block}  inside_subblock = {inside_subblock}   current_subblock = {current_subblock}   current_block = {current_block}   sub_block_id = {sub_block_id}")
#             if not inside_env:
#                 continue
#             if not inside_block:
#                 continue
#             if sub_block_id != "default" and not inside_subblock:
#                 continue

#             m = pattern.match(line)
#             if m:
#                 lhs = m.group(1)
#                 rhs = m.group(2).strip()

#                 if rhs.startswith('"') and rhs.endswith('"'):
#                     new_rhs = f'"{new_val}"'
#                 elif rhs.startswith("'") and rhs.endswith("'"):
#                     new_rhs = f"'{new_val}'"
#                 else:
#                     new_rhs = str(new_val)

#                 updated_lines[i] = f"{lhs}{new_rhs}\n"
#                 replaced = True
#                 print(f"Updated {lhs_var} in {env_update}/{block_id}/{sub_block_id} → {new_rhs} (line {i+1})")
#                 break

#         if not replaced:
#             print(f"Could not update {lhs_var}: no matching env/block/subblock found")

#     with open(terraform_file, "w") as f:
#         f.writelines(updated_lines)

# def update_terraform_file_locals_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     # --- Load Terraform locals file ---
#     with open(terraform_file, "r") as f:
#         lines = f.readlines()
#     updated_lines = lines[:]

#     # --- Load metadata JSON ---
#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)

#     # --- Gather all known environments ---
#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])

#     assign_pattern_template = r"^(\s*{}\s*=\s*)(.*)"

#     # --- Process each update ---
#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

#         # --- Find best matching metadata variable (filter by environment first) ---
#         sizing_vars = metadata.get("sizing_variables", [])
#         same_env_vars = [v for v in sizing_vars if v.get("environment") == env_update]
#         search_space = same_env_vars if same_env_vars else sizing_vars
#         if not same_env_vars:
#             print(f"WARNING: No metadata entries found for environment '{env_update}', falling back to all environments.")

#         best_match = None
#         best_score = -1
#         for var in search_space:
#             score = similarity(target_var.lower(), var.get("id", "").lower())
#             if score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env={env_update} (best score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")
#         env_meta = best_match.get("environment")

#         pattern = re.compile(assign_pattern_template.format(re.escape(lhs_var)))

#         replaced = False
#         brace_depth = 0
#         inside_env = inside_block = inside_subblock = False
#         current_env = current_block = current_subblock = None

#         # --- Iterate through file lines ---
#         for i, line in enumerate(updated_lines):
#             # Skip comments
#             if re.match(r'^\s*#', line) or re.match(r'^\s*/\*', line):
#                 continue

#             # Count braces before context matching
#             open_braces = line.count("{")
#             close_braces = line.count("}")
#             brace_depth += open_braces - close_braces

#             # --- Detect Environment ---
#             env_match = re.match(r'^\s*(\w+)\s*=\s*{', line)
#             if env_match and brace_depth == 1:
#                 env_name = env_match.group(1)
#                 if env_name in known_envs:
#                     current_env = env_name
#                     inside_env = (env_name == env_update)
#                     print(f"DEBUG: Entering env {current_env} at line {i+1}")

#             # --- Detect Block (service/resource) ---
#             block_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
#             if block_match and inside_env and brace_depth == 2:
#                 blk_name = block_match.group(1)
#                 current_block = blk_name
#                 inside_block = (blk_name == block_id)
#                 print(f"DEBUG: Inside block {current_block} at line {i+1} (inside_block={inside_block})")

#             # --- Detect Sub-block (instance) ---
#             subblock_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
#             if subblock_match and inside_block and brace_depth == 3:
#                 subblk = subblock_match.group(1)
#                 current_subblock = subblk
#                 inside_subblock = (subblk == sub_block_id)
#                 print(f"DEBUG: Inside sub-block {current_subblock} at line {i+1} (inside_subblock={inside_subblock})")

#             # --- Apply Update ---
#             if inside_env and inside_block and (sub_block_id == "default" or inside_subblock):
#                 m = pattern.match(line)
#                 if m:
#                     lhs = m.group(1)
#                     rhs = m.group(2).strip()

#                     # Preserve quote type
#                     if rhs.startswith('"') and rhs.endswith('"'):
#                         new_rhs = f'"{new_val}"'
#                     elif rhs.startswith("'") and rhs.endswith("'"):
#                         new_rhs = f"'{new_val}'"
#                     else:
#                         new_rhs = str(new_val)

#                     updated_lines[i] = f"{lhs}{new_rhs}\n"
#                     replaced = True
#                     print(f"Updated {lhs_var} in {env_update}/{block_id}/{sub_block_id} → {new_rhs} (line {i+1})")
#                     break

#         if not replaced:
#             print(f" Could not update {lhs_var}: no matching env/block/subblock found")

    # with open(terraform_file, "w") as f:
    #     f.writelines(updated_lines)

# def update_terraform_file_locals_type(terraform_file: str, metadata_file: str, updates: list, similarity_threshold: float = 0.75):
#     with open(terraform_file, "r") as f:
#         lines = f.readlines()
#     updated_lines = lines[:]

#     with open(metadata_file, "r") as f:
#         metadata = json.load(f)

#     known_envs = set()
#     for var in metadata.get("sizing_variables", []):
#         if var.get("environment"):
#             known_envs.add(var["environment"])

#     assign_pattern_template = r"^(\s*{}\s*=\s*)(.*)"

#     for update_item in updates:
#         env_update = update_item["environment_name"]
#         rg_update = update_item["resource_group_name"]
#         rn_update = update_item["resource_name"]
#         target_var = update_item["variable_name"]
#         new_val = update_item["variable_value"]

#         print(f"\n=== Processing update: {target_var} for {env_update}/{rg_update}/{rn_update} → {new_val} ===")

#         sizing_vars = metadata.get("sizing_variables", [])
#         same_env_vars = [v for v in sizing_vars if v.get("environment") == env_update]
#         search_space = same_env_vars if same_env_vars else sizing_vars

#         best_match = None
#         best_score = -1
#         for var in search_space:
#             score = similarity(target_var.lower(), var.get("id", "").lower())
#             if score > best_score:
#                 best_score = score
#                 best_match = var

#         if not best_match or best_score < similarity_threshold:
#             print(f"No good metadata match for {target_var} in env={env_update} (best score={best_score:.2f})")
#             continue

#         lhs_var = best_match["id"]
#         block_id = best_match.get("service")
#         sub_block_id = best_match.get("instance_id")

#         pattern = re.compile(assign_pattern_template.format(re.escape(lhs_var)))

#         replaced = False
#         brace_depth = 0
#         locals_depth_offset = None
#         inside_env = inside_block = inside_subblock = False
#         current_env = current_block = current_subblock = None

#         for i, line in enumerate(updated_lines):
#             if re.match(r'^\s*#', line) or re.match(r'^\s*/\*', line):
#                 continue

#             # Detect start of locals block
#             if locals_depth_offset is None and re.search(r'^\s*locals\s*=\s*{|^\s*locals\s*{', line):
#                 locals_depth_offset = brace_depth + 1  # environment starts one level deeper
#                 print(f"DEBUG: Found top-level locals at line {i+1}, setting offset={locals_depth_offset}")

#             open_braces = line.count("{")
#             close_braces = line.count("}")
#             brace_depth += open_braces - close_braces

#             # Adjusted depth
#             effective_depth = brace_depth - (locals_depth_offset or 0)

#             # Detect environment
#             env_match = re.match(r'^\s*(\w+)\s*=\s*{', line)
#             if env_match and effective_depth == 1:
#                 env_name = env_match.group(1)
#                 if env_name in known_envs:
#                     current_env = env_name
#                     inside_env = (env_name == env_update)
#                     print(f"DEBUG: Entering env {current_env} at line {i+1}")

#             # Detect block
#             block_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
#             if block_match and inside_env and effective_depth == 2:
#                 blk_name = block_match.group(1)
#                 current_block = blk_name
#                 inside_block = (blk_name == block_id)
#                 print(f"DEBUG: Inside block {current_block} at line {i+1} (inside_block={inside_block})")

#             # Detect sub-block
#             subblock_match = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*{', line)
#             if subblock_match and inside_block and effective_depth == 3:
#                 subblk = subblock_match.group(1)
#                 current_subblock = subblk
#                 inside_subblock = (subblk == sub_block_id)
#                 print(f"DEBUG: Inside sub-block {current_subblock} at line {i+1} (inside_subblock={inside_subblock})")

#             # Apply update
#             if inside_env and inside_block and (sub_block_id == "default" or inside_subblock):
#                 m = pattern.match(line)
#                 if m:
#                     lhs = m.group(1)
#                     rhs = m.group(2).strip()
#                     if rhs.startswith('"') and rhs.endswith('"'):
#                         new_rhs = f'"{new_val}"'
#                     elif rhs.startswith("'") and rhs.endswith("'"):
#                         new_rhs = f"'{new_val}'"
#                     else:
#                         new_rhs = str(new_val)
#                     updated_lines[i] = f"{lhs}{new_rhs}\n"
#                     replaced = True
#                     print(f"Updated {lhs_var} in {env_update}/{block_id}/{sub_block_id} → {new_rhs} (line {i+1})")
#                     break

#         if not replaced:
#             print(f"Could not update {lhs_var}: no matching env/block/subblock found")

#     with open(terraform_file, "w") as f:
#         f.writelines(updated_lines)


def main():
    # terraform_file_locals_type = "./local_astratest-Azure-Cld3-MACS-ent-member-ss-api-main.tf"
    # metadata_file_locals_type = "./analysis_astratest-Azure-Cld3-MACS-ent-member-ss-api-main.json"
    # finops_data_updates_locals_type = [
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "macs-ent-member-ss-api-eastus2-npe-rg",
    #  "resource_name": "ent-member",
    #  "variable_name": "vm-size",
    #  "variable_value": "Standard_DOoooD2s_v3111"
    # },
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "macs-ent-member-ss-api-eastus2-npe-rg",
    #  "resource_name": "ent-member",
    #  "variable_name": "os-disk-size",
    #  "variable_value": "10000027"
    # },
    # {
    #  "environment_name": "prd",
    #  "resource_group_name": "macs-ent-member-ss-api-eastus2-prd-rg",
    #  "resource_name": "epssmicroapi-eastus2-prd",
    #  "variable_name": "capacity",
    #  "variable_value": 444
    # }
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "macs-ent-member-ss-api-eastus2-prd-rg",
    #  "resource_name": "epssmicroapi-eastus2-prd",
    #  "variable_name": "data-disk-size-list",
    #  "variable_value": ["Test_LRS", "Test_LRS", "Test_LRS"]
    # }
    # ] 

    # terraform_file_locals_type = "./local_AstraTestRepo-main.tf"
    # metadata_file_locals_type = "./analysis_AstraTestRepo-main.json"
    # finops_data_updates_locals_type = [
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "mgmt-spoke-dba-eastus2-npe-rg",
    #  "resource_name": "astra-linuxvm-test01",
    #  "variable_name": "vm-size",
    #  "variable_value": "Standard_AAAAs_v0111"
    # },
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "mgmt-spoke-dba-eastus2-npe-rg",
    #  "resource_name": "astra-linuxvm-test01",
    #  "variable_name": "os-disk-size",
    #  "variable_value": "10000027xxyy"
    # }   
    # ]

    # terraform_file_locals_type = "./local_astratest-Azure-Cld3-Shared-Services-EPP-Nabu_Postgresql-main.tf"
    # metadata_file_locals_type = "./analysis_astratest-Azure-Cld3-Shared-Services-EPP-Nabu_Postgresql-main.json"
    # finops_data_updates_locals_type = [
    # {
    #  "environment_name": "prod",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "postgre-db-replica-count",
    #  "variable_value": 100
    # },
    # {
    #  "environment_name": "prod",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "postgre-db-server-tier",
    #  "variable_value": "SpecialPurpose"
    # },
    # {
    #  "environment_name": "npe",
    #  "resource_group_name": "az3-epp-nabu-eastus2-npe-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-npe-rg",
    #  "variable_name": "postgre-db-storage",
    #  "variable_value": 15555
    # },   
    # ]

    # terraform_file_locals_type = "./local_east_Azure-Cld3-SHARED001-strongdm-main.tf"
    # metadata_file_locals_type = "./analysis_Azure-Cld3-SHARED001-strongdm-main.json"
    # finops_data_updates_locals_type = [
    # {
    #  "environment_name": "deveast",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "storage-account-tier",
    #  "variable_value": "Non-Standard"
    # },
    # {
    #  "environment_name": "deveast",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "access-tier",
    #  "variable_value": "Cold"
    # },
    # {
    #  "environment_name": "prdeast",
    #  "resource_group_name": "shared001-strongdm-eastus2-prod-rg",
    #  "resource_name": "shared001-strongdm-eastus2-prod-rg",
    #  "variable_name": "storage-replication-type",
    #  "variable_value": "MMRSSS"
    # },   
    # {
    #  "environment_name": "prdeast",
    #  "resource_group_name": "shared001-strongdm-eastus2-prod-rg",
    #  "resource_name": "shared001-strongdm-eastus2-prod-rg",
    #  "variable_name": "file-share-quota",
    #  "variable_value": 111
    # },
    # ]

    # terraform_file_locals_type = "./local_east_Azure-Cld3-SHARED001-strongdm-main.tf"
    # metadata_file_locals_type = "./analysis_Azure-Cld3-SHARED001-strongdm-main.json"
    # finops_data_updates_locals_type = [
    # {
    #  "environment_name": "deveast",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "storage-account-tier",
    #  "variable_value": "Non-Standard"
    # },
    # {
    #  "environment_name": "deveast",
    #  "resource_group_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "resource_name": "az3-epp-nabu-eastus2-prd-rg",
    #  "variable_name": "access-tier",
    #  "variable_value": "Cold"
    # },
    # {
    #  "environment_name": "prdeast",
    #  "resource_group_name": "shared001-strongdm-eastus2-prod-rg",
    #  "resource_name": "shared001-strongdm-eastus2-prod-rg",
    #  "variable_name": "storage-replication-type",
    #  "variable_value": "MMRSSS"
    # },   
    # {
    #  "environment_name": "prdeast",
    #  "resource_group_name": "shared001-strongdm-eastus2-prod-rg",
    #  "resource_name": "shared001-strongdm-eastus2-prod-rg",
    #  "variable_name": "file-share-quota",
    #  "variable_value": 111
    # },
    # ]

    terraform_file_yaml_type = "./yaml_astratest-Azure-Cld3-DIGE-azuresql-main.yaml"
    metadata_file_yaml_type = "./analysis_astratest-Azure-Cld3-DIGE-azuresql-main.json"    
    finops_data_updates_yaml_type = [
    {
     "environment_name": "dev",
     "resource_group_name": "dige-azuresql-eastus2-dev-rg",   
     "resource_name": "dhp-pisp-sqldb-plat-profile",   
     "variable_name": "max-size-gb",
     "variable_value": 1000
    },    
    {
     "environment_name": "dev",
     "resource_group_name": "dige-azuresql-eastus2-dev-rg",   
     "resource_name": "dhp-pisp-sqldb-plat-profile",   
     "variable_name": "min-capacity",
     "variable_value": 0
    },
    {
     "environment_name": "prod",
     "resource_group_name": "dige-azuresql-eastus2-dev-rg",   
     "resource_name": "dhp-pisp-sqldb-plat-profile",   
     "variable_name": "max-size-gb",
     "variable_value": 5000
    }        
    ]

    # terraform_file_tfvars_type = "./autotfvars_astratest-Azure-Cld3-Shared-Services-PlainID_redis-centralus.auto.tfvars"
    # finops_data_updates_tfvars_type = [
    # {
    #  "resource_group_name": "az3-plainid-eastus2-prd-rg",
    #  "resource_name": "az3-plainid-eastus2-prd-redis",
    #  "variable_name": "capacity",
    #  "variable_value": 10
    # },
    # {
    #  "resource_group_name": "az3-plainid-eastus2-prd-rg",
    #  "resource_name": "az3-plainid-eastus2-prd-redis",
    #  "variable_name": "sku-name",
    #  "variable_value": "Non-Standard"
    # }
    # ]

    # update_terraform_file_locals_type(terraform_file_locals_type, metadata_file_locals_type, finops_data_updates_locals_type, similarity_threshold=0.75)   # Tune similarity threshold here
    update_terraform_file_yaml_type(terraform_file_yaml_type, metadata_file_yaml_type, finops_data_updates_yaml_type, similarity_threshold=0.75)   # Tune similarity threshold here
    # update_terraform_file_tfvars_type(terraform_file_tfvars_type, finops_data_updates_tfvars_type, similarity_threshold=0.55)   # Tune similarity threshold here


if __name__ == "__main__":
    main()