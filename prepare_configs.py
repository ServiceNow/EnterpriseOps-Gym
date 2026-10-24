#!/usr/bin/env python3
"""
Convert delivered task-authoring JSON (task_id/uuid/config wrapper, with
scenario_config/verifier_configs/gym_server_id fields) into the flat
BenchmarkConfig-shaped JSON that evaluate.py's --configs_folder expects.

Usage:
    python3 prepare_configs.py \
        --input_dir delivery_batch_09-22 \
        --seed_dir delivery_batch_09-22/db_seeds \
        --output_dir configs_folder \
        --mode oracle --domain csm
"""

import argparse
import glob
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_CONF_PATH = os.path.join(PROJECT_ROOT, "conf", "ray", "domain_conf.json")


def load_domain_conf(path: str = DOMAIN_CONF_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def convert_task(task_json: dict, seed_dir: str, seed_files: list, domain_conf: dict) -> dict:
    """Convert one delivered task dict into a flat BenchmarkConfig dict.

    Raises ValueError if the task cannot be converted (e.g. no unique
    seed-file match for a gym's database_id).
    """
    cfg = task_json["config"]
    sc = cfg["scenario_config"]
    gym_id_to_mcp_name = {k: v["MCP_NAME"] for k, v in domain_conf.items()}

    gym_servers_config = []
    # Multi-domain (hybrid) tasks name tools "<gym_server_id>__<tool>" so two
    # servers can expose the same tool name. A server gets a tool prefix only
    # when the task itself uses that prefix; single-domain tasks keep bare names.
    task_tool_names = [
        *(sc.get("selected_tools") or []),
        *(sc.get("expected_tools") or []),
        *(step.get("tool_name", "") for step in cfg.get("manual_tool_executions") or []),
    ]
    for g in cfg["gym_servers_config"]:
        database_id = g.get("database_id")
        if not database_id:
            raise ValueError("gym_servers_config entry has no database_id")
        # Seed files are named "<database_id>_<label>_seed.sql". Matching on the
        # trailing separator keeps one id from matching another that extends it.
        matches = [s for s in seed_files if s.startswith(f"{database_id}_")]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one seed file for database_id={database_id!r}, "
                f"found {len(matches)}: {sorted(matches)}"
            )
        gym_id = g["gym_server_id"]
        if gym_id not in domain_conf:
            raise ValueError(f"unknown gym_server_id={gym_id!r} (not in domain_conf.json)")
        domain_info = domain_conf[gym_id]
        server = {
            "mcp_server_name": domain_info["MCP_NAME"],
            "mcp_server_url": domain_info["MCP_ENDPOINT"],
            "seed_database_file": os.path.join(seed_dir, matches[0]),
            "context": g.get("context", {}),
        }
        if any(name.startswith(f"{gym_id}__") for name in task_tool_names):
            server["tool_prefix"] = gym_id
        gym_servers_config.append(server)

    verifiers = []
    for v in cfg["verifier_configs"]:
        target_gym = v.get("target_gym_server")
        verifiers.append(
            {
                "verifier_type": v["type"],
                "name": v.get("name"),
                "description": v.get("description"),
                # target_gym_server refers to a gym_server_id (e.g. "csm"); verifiers
                # are matched against gym_servers_config by mcp_server_name at runtime,
                # so this must go through the same domain_conf lookup as the gym config.
                "gym_name": gym_id_to_mcp_name.get(target_gym, target_gym),
                "validation_config": {
                    "query": v.get("sql_query"),
                    "expected_value": v.get("expected_value"),
                    "comparison_type": v.get("comparison_type", "equals"),
                },
            }
        )

    return {
        "task_id": cfg["task_id"],
        "system_prompt": sc["system_prompt"],
        "user_prompt": sc["prompt"],
        "gym_servers_config": gym_servers_config,
        "verifiers": verifiers,
        "number_of_runs": 1,
        "selected_tools": sc.get("selected_tools"),
        "restricted_tools": sc.get("restricted_tools"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory of delivered task_*.json files (task_id/uuid/config wrapper).",
    )
    parser.add_argument(
        "--seed_dir", default=None,
        help="Directory of <database_id>_*.sql seed files. Defaults to <input_dir>/db_seeds.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write flat configs into (matches evaluate.py's --configs_folder).",
    )
    parser.add_argument("--mode", default="oracle", help="Mode label used in output filenames.")
    parser.add_argument("--domain", default="csm", help="Domain label used in output filenames.")
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing *.json files in --output_dir before writing.",
    )
    args = parser.parse_args()

    seed_dir = os.path.abspath(args.seed_dir or os.path.join(args.input_dir, "db_seeds"))
    seed_files = os.listdir(seed_dir)
    domain_conf = load_domain_conf()

    task_files = sorted(glob.glob(os.path.join(args.input_dir, "task_*.json")))
    if not task_files:
        raise SystemExit(f"No task_*.json files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.clean:
        for f in glob.glob(os.path.join(args.output_dir, "*.json")):
            os.remove(f)

    written, skipped = 0, 0
    for tf in task_files:
        with open(tf, "r") as f:
            task_json = json.load(f)
        task_id = task_json["config"]["task_id"]
        try:
            flat_config = convert_task(task_json, seed_dir, seed_files, domain_conf)
        except (KeyError, ValueError, TypeError) as e:
            print(f"SKIP {os.path.basename(tf)}: {e}")
            skipped += 1
            continue

        out_path = os.path.join(
            args.output_dir, f"{args.mode}__{args.domain}__{task_id}.json"
        )
        with open(out_path, "w") as f:
            json.dump(flat_config, f, indent=2)
        written += 1

    print(f"Wrote {written} configs to {args.output_dir} ({skipped} skipped)")
    if skipped:
        # A silently shorter config set means a silently shorter benchmark.
        print(
            f"WARNING: {skipped} of {len(task_files)} task(s) were not converted; "
            f"the evaluation will cover fewer tasks than delivered."
        )


if __name__ == "__main__":
    main()
