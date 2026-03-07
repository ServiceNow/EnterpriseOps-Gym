#!/usr/bin/env python3
"""
Standalone Benchmark Executor
Replicates the complete /create and /execute benchmark API behavior without database dependencies.
Relies only on config.json for all configuration.

This script implements the full benchmark execution flow:
1. Load MCP tools from server
2. Send system + user prompts to LLM
3. LLM decides which MCP tools to call
4. Execute MCP tool calls via JSON-RPC
5. Send tool results back to LLM
6. Repeat until task completion
7. Run verifiers (database_state, response_checker)

Requirements:
- Python 3.9+
- config.json in same directory
- LangChain, httpx, anthropic/openai libraries
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import random
import tempfile
from datasets import load_dataset as hf_load_dataset

from benchmark.executor import BenchmarkExecutor
from benchmark.models import BenchmarkConfig
from benchmark_utils import load_llm_configs, skip_sample
from utils.task_queue_worker import TaskQueueWorker
from orchestrators.react import ReactOrchestrator
from orchestrators.planner_react import PlannerReactOrchestrator
from orchestrators.decomposing_planner import DecomposingPlannerOrchestrator

ORCHESTRATOR_MAP = {
    "react": ReactOrchestrator,
    "planner_react": PlannerReactOrchestrator,
    "decomposing": DecomposingPlannerOrchestrator,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================


def load_config(config_path: str = "config.json") -> BenchmarkConfig:
    """
    Load configuration from config.json.
    Supports both single-gym and multi-gym formats.

    Single-gym format (legacy):
    {
        "mcp_server_url": "...",
        "database_id": "...",
        ...
    }

    Multi-gym format (new):
    {
        "gym_servers_config": [
            {
                "mcp_server_name": "...",
                "mcp_server_url": "...",
                "database_id": "..."  // optional
            }
        ],
        ...
    }
    """
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Remove comment fields (any key starting with underscore)
        config_data = {k: v for k, v in config_data.items() if not k.startswith("_")}

        # Clean verifiers - remove _description fields
        if "verifiers" in config_data and config_data["verifiers"]:
            cleaned_verifiers = []
            for verifier in config_data["verifiers"]:
                cleaned_verifier = {
                    k: v for k, v in verifier.items() if not k.startswith("_")
                }
                cleaned_verifiers.append(cleaned_verifier)
            config_data["verifiers"] = cleaned_verifiers

        # Determine configuration type
        has_single_gym = "mcp_server_url" in config_data
        has_multi_gym = "gym_servers_config" in config_data

        if has_single_gym and has_multi_gym:
            logger.warning(
                "⚠️  Both single-gym and multi-gym configurations found. "
                "Multi-gym configuration will be used."
            )
        elif not has_single_gym and not has_multi_gym:
            raise ValueError(
                "Configuration must include either 'mcp_server_url' (single-gym) "
                "or 'gym_servers_config' (multi-gym)"
            )

        # Validate multi-gym configuration if provided
        if has_multi_gym:
            gym_servers = config_data.get("gym_servers_config", [])

            if not isinstance(gym_servers, list):
                raise ValueError("'gym_servers_config' must be a list")

            if len(gym_servers) == 0:
                raise ValueError("'gym_servers_config' cannot be empty")

            # Validate each gym server config
            for idx, gym_config in enumerate(gym_servers):
                if not isinstance(gym_config, dict):
                    raise ValueError(f"gym_servers_config[{idx}] must be a dictionary")

                required_fields = ["mcp_server_name", "mcp_server_url"]
                for field in required_fields:
                    if field not in gym_config:
                        raise ValueError(
                            f"gym_servers_config[{idx}] missing required field: '{field}'"
                        )

                # database_id is optional (can be auto-created)
                if "database_id" in gym_config and gym_config["database_id"]:
                    logger.info(
                        f"  Gym '{gym_config['mcp_server_name']}': "
                        f"database_id = {gym_config['database_id']}"
                    )
                else:
                    logger.info(
                        f"  Gym '{gym_config['mcp_server_name']}': "
                        f"database_id will be auto-created"
                    )

            logger.info(
                f"✅ Multi-gym configuration validated: {len(gym_servers)} gym(s)"
            )

            # For multi-gym, required fields at top level (not gym-specific)
            required_fields = [
                "system_prompt",
                "user_prompt",
            ]
        else:
            # Single-gym validation
            required_fields = [
                "mcp_server_url",
                "database_id",
                "system_prompt",
                "user_prompt",
            ]

        for field in required_fields:
            if field not in config_data:
                raise ValueError(f"Missing required field in config.json: {field}")

        # Set defaults
        config_data.setdefault("mcp_endpoint", "/mcp")
        config_data.setdefault("verifiers", [])
        config_data.setdefault("number_of_runs", 1)
        config_data.setdefault("context", {})
        config_data.setdefault("temperature", 0.6)
        config_data.setdefault("max_tokens", 16384)
        logger.info("✅ Configuration loaded successfully")
        logger.info(config_data)

        return BenchmarkConfig(**config_data)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


async def execute_sample(
    config_file, llm_config, output_folder,
    orchestrator="react", planner_llm_config=None, max_num_attempts=5
):
    if skip_sample(config_file, output_folder):
        print(f"Skipping already processed config: {config_file}")
        return
    print(f"Running benchmark for config: {config_file}")
    config = load_config(config_file)
    llm_config = random.choice(
        load_llm_configs(llm_config)
    )  # MARKER: Load balancer picks a random LLM instance
    try:
        logger.info(f"Using LLM config: {llm_config.llm_api_endpoint}")
    except Exception:
        logger.info(f"Failed to log LLM config endpoint.")

    orchestrator_class = ORCHESTRATOR_MAP[orchestrator]
    orchestrator_kwargs = {}
    if planner_llm_config is not None:
        orchestrator_kwargs["planner_llm_config"] = random.choice(
            load_llm_configs(planner_llm_config)
        )

    executor = BenchmarkExecutor(
        config,
        llm_config=llm_config,
        orchestrator_class=orchestrator_class,
        orchestrator_kwargs=orchestrator_kwargs,
        config_path=config_file,
    )

    result = None
    for i in range(max_num_attempts):
        result = await executor.execute_benchmark()
        error = any(
            [run.get("error") for run in result["runs"]]
        )  # If any of the runs fails, we retry the full sample.
        if not error:
            break
        print(f"Attempt {i+1} failed with error: {error}")
        if i < max_num_attempts - 1:
            await asyncio.sleep(i + 1)  # Linear backoff
        else:
            print("Max attempts reached. Saving last result with error.")

    output_file = os.path.join(
        output_folder,
        f"results_{os.path.basename(config_file).replace('.json', '')}.json",
    )
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_folder", type=str, default=None,
                        help="Local folder containing task JSON configs.")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset repo ID (e.g. ServiceNow-AI/EnterpriseOps-Gym). "
                             "Requires --domain and --mode.")
    parser.add_argument("--domain", type=str, nargs="+", default=None,
                        help="One or more domains to evaluate (e.g. teams csm). Used with --hf_dataset.")
    parser.add_argument("--mode", type=str, nargs="+", default=["oracle"],
                        help="One or more tool-set modes (e.g. oracle +5_tools). Used with --hf_dataset.")
    parser.add_argument("--llm_config", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--num_runs",
        type=str,
        default=3,
        help="Number of runs to execute.",
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default="react",
        choices=["react", "planner_react", "decomposing"],
        help="Orchestration strategy.",
    )
    parser.add_argument(
        "--planner_llm_config",
        type=str,
        default=None,
        help="Path to LLM config for the planner (required for planner_react and decomposing).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    if args.hf_dataset:
        if not args.domain:
            raise ValueError("--domain is required when using --hf_dataset")
        domains = args.domain
        modes = args.mode
        tmp_dir = tempfile.mkdtemp(prefix="rl_gym_hf_")
        json_string_fields = {"gym_servers_config", "verifiers"}
        hf_only_fields = {"task_id", "domain"}
        total_written = 0
        for mode in modes:
            for domain in domains:
                logger.info(
                    f"Loading configs from HuggingFace: {args.hf_dataset} "
                    f"(config={mode}, split={domain})"
                )
                hf_ds = hf_load_dataset(args.hf_dataset, mode, split=domain)
                for row in hf_ds:
                    task_id = row.get("task_id", f"task_{id(row)}")
                    file_name = f"{mode}__{domain}__{task_id}.json"
                    task_dict = {}
                    for k, v in row.items():
                        if k in hf_only_fields:
                            continue
                        if k in json_string_fields and isinstance(v, str):
                            v = json.loads(v)
                        task_dict[k] = v
                    with open(os.path.join(tmp_dir, file_name), "w") as f:
                        json.dump(task_dict, f)
                    total_written += 1
        configs_folder = tmp_dir
        logger.info(f"Wrote {total_written} task configs to temp dir: {tmp_dir}")
    else:
        if not args.configs_folder:
            raise ValueError("Either --configs_folder or --hf_dataset must be provided")
        configs_folder = args.configs_folder

    config_files = glob.glob(os.path.join(configs_folder, "*.json"))
    for idx in range(int(args.num_runs)):
        output_folder = os.path.join(args.output_folder, f"run_{idx+1}")
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Processing {len(config_files)} config files with concurrency {args.concurrency} into folder: {output_folder}")
        worker = TaskQueueWorker(
            worker_method=lambda cfg: execute_sample(
                cfg, args.llm_config, output_folder,
                orchestrator=args.orchestrator,
                planner_llm_config=args.planner_llm_config,
            ),
            concurrency=int(args.concurrency),
        )
        await worker.process(config_files)


if __name__ == "__main__":
    asyncio.run(main())
