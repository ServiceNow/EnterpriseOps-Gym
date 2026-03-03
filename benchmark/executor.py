"""
BenchmarkExecutor: Orchestrates the complete benchmark execution flow.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

from benchmark.llm_client import LLMClient
from benchmark.mcp_client import MCPClient, create_database_from_file, delete_database
from benchmark.models import BenchmarkConfig, LLMConfig, VerifierConfig
from benchmark.verifier import VerifierEngine

logger = logging.getLogger(__name__)


# ============================================================================
# BENCHMARK EXECUTOR
# ============================================================================


class BenchmarkExecutor:
    """
    Main benchmark executor that orchestrates the complete flow:
    1. Load MCP tools from multiple gyms
    2. Send prompts to LLM with merged tools
    3. Execute tool calls (routed to correct gym)
    4. Loop until completion
    5. Run verifiers

    Supports both single-gym (legacy) and multi-gym configurations.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        llm_config: LLMConfig,
        orchestrator_class=None,
        orchestrator_kwargs=None,
        config_path: str = "config.json",
    ):
        if orchestrator_class is None:
            from orchestrators.react import ReactOrchestrator
            orchestrator_class = ReactOrchestrator
        self.orchestrator_class = orchestrator_class
        self.orchestrator_kwargs = orchestrator_kwargs or {}
        self.config = config
        self.config_path = config_path  # Store for resolving relative paths
        self.llm_config = llm_config  # Store LLM config
        self.mcp_clients = {}  # Map of gym_name -> MCPClient
        self.llm_client = None
        self.verifier_engine = None
        self.available_tools = []  # Merged tools from all gyms
        self.tool_to_server_mapping = {}  # Maps tool_name -> gym_name
        self.gym_configs = []  # List of gym server configurations
        self.auto_created_databases = []  # Track auto-created databases for cleanup

    async def initialize(self):
        """Initialize all clients for multi-gym support"""
        logger.info("Initializing benchmark executor...")

        # Parse gym configurations only if not already parsed (support both multi-gym and legacy single-gym)
        # Note: execute_benchmark() pre-populates self.gym_configs with database_ids
        if not self.gym_configs:
            self.gym_configs = self._parse_gym_configs()

        logger.info(f"📋 Configured {len(self.gym_configs)} gym server(s)")
        for idx, gym_config in enumerate(self.gym_configs):
            logger.info(
                f"  Gym #{idx + 1}: {gym_config['mcp_server_name']} -> {gym_config['mcp_server_url']}"
            )
            logger.info(f"    Database ID: {gym_config['database_id']}")

        # Initialize MCP clients for each gym
        for gym_config in self.gym_configs:
            gym_name = gym_config["mcp_server_name"]

            client = MCPClient(
                base_url=gym_config["mcp_server_url"],
                auth_config=gym_config.get("auth_config"),
                mcp_endpoint=gym_config.get("mcp_endpoint", "/mcp"),
                database_id=gym_config["database_id"],
                context=gym_config.get("context", {}),
            )

            connected = await client.connect()
            if not connected:
                raise Exception(f"Failed to connect to MCP server: {gym_name}")

            self.mcp_clients[gym_name] = client
            logger.info(f"✅ Connected to gym: {gym_name}")

        # Discover and merge tools from all gyms
        await self._discover_and_merge_tools()

        # Apply tool restrictions if configured
        if self.config.restricted_tools:
            original_count = len(self.available_tools)
            self.available_tools = [
                tool
                for tool in self.available_tools
                if tool["name"] not in self.config.restricted_tools
            ]
            logger.info(
                f"Applied tool restrictions: {original_count} -> {len(self.available_tools)} tools"
            )

        # Initialize LLM client
        self.llm_client = LLMClient(
            self.llm_config.llm_provider,
            self.llm_config.llm_model,
            self.llm_config.llm_api_key,
            api_endpoint=self.llm_config.llm_api_endpoint,
            api_version=self.llm_config.llm_api_version,
            region=self.llm_config.llm_region,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            top_p=self.llm_config.top_p,
            effort=self.llm_config.effort,
        )

        # Initialize verifier engine (multi-gym aware)
        # Pass all MCP clients so verifiers can query the correct gym database
        self.verifier_engine = VerifierEngine(self.mcp_clients, self.llm_client)

        # If orchestrator_kwargs contains a raw planner_llm_config, convert it to an
        # initialized LLMClient (with retry) so orchestrators receive planner_llm_client.
        if "planner_llm_config" in self.orchestrator_kwargs:
            planner_llm_config = self.orchestrator_kwargs.pop("planner_llm_config")
            self._initialize_planner_llm(planner_llm_config)
            self.orchestrator_kwargs["planner_llm_client"] = self.planner_llm_client

        logger.info("✅ Initialization complete")

    def _initialize_planner_llm(self, planner_llm_config: "LLMConfig") -> None:
        """Initialize planner LLM client with retry. Sets self.planner_llm_client."""
        logger.info("Initializing planner LLM...")
        self.planner_llm_client = LLMClient(
            provider=planner_llm_config.llm_provider,
            model=planner_llm_config.llm_model,
            api_key=planner_llm_config.llm_api_key,
            api_endpoint=planner_llm_config.llm_api_endpoint,
            api_version=planner_llm_config.llm_api_version,
            region=planner_llm_config.llm_region,
            temperature=planner_llm_config.temperature,
            max_tokens=planner_llm_config.max_tokens,
            top_p=planner_llm_config.top_p,
            effort=planner_llm_config.effort,
        )
        self.planner_llm_client.llm = self.planner_llm_client.llm.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=3,
        )
        logger.info(f"✅ Planner initialized: {planner_llm_config.llm_provider}/{planner_llm_config.llm_model}")

    def _parse_gym_configs(self) -> List[Dict[str, Any]]:
        """Parse gym configurations from config, supporting both multi-gym and legacy formats"""

        # Multi-gym configuration (preferred)
        if self.config.gym_servers_config:
            logger.info("Using multi-gym configuration")
            gym_configs = []

            for idx, server_config in enumerate(self.config.gym_servers_config):
                # Validate required fields
                if "mcp_server_name" not in server_config:
                    raise ValueError(
                        f"gym_servers_config[{idx}] missing 'mcp_server_name'"
                    )
                if "mcp_server_url" not in server_config:
                    raise ValueError(
                        f"gym_servers_config[{idx}] missing 'mcp_server_url'"
                    )

                # seed_database_file is optional - will use api/sample-data if empty
                gym_config = {
                    "mcp_server_name": server_config["mcp_server_name"],
                    "mcp_server_url": server_config["mcp_server_url"],
                    "seed_database_file": server_config.get("seed_database_file", ""),
                    "database_id": "",  # Will be populated at runtime
                    "mcp_endpoint": server_config.get("mcp_endpoint", "/mcp"),
                    "auth_config": server_config.get("auth_config"),
                    "context": server_config.get("context", {}),
                }
                gym_configs.append(gym_config)

            return gym_configs

        # Legacy single-gym configuration (backward compatibility)
        elif self.config.mcp_server_url:
            logger.info("Using legacy single-gym configuration")
            return [
                {
                    "mcp_server_name": "default_gym",
                    "mcp_server_url": self.config.mcp_server_url,
                    "seed_database_file": self.config.get("seed_database_file", ""),
                    "database_id": "",  # Will be populated at runtime
                    "mcp_endpoint": self.config.mcp_endpoint or "/mcp",
                    "auth_config": self.config.auth_config,
                    "context": self.config.context or {},
                }
            ]

        else:
            raise ValueError(
                "No gym configuration found. Provide either 'gym_servers_config' (multi-gym) "
                "or 'mcp_server_url' (legacy single-gym)"
            )

    async def _discover_and_merge_tools(self):
        """Discover tools from all gyms and merge them with server routing information"""
        logger.info(f"\n{'='*80}")
        logger.info("DISCOVERING TOOLS FROM ALL GYMS")
        logger.info(f"{'='*80}\n")

        merged_tools = []
        tool_to_server_mapping = {}

        for idx, gym_config in enumerate(self.gym_configs):
            gym_name = gym_config["mcp_server_name"]
            client = self.mcp_clients[gym_name]

            logger.info(f"[TOOL_SOURCE] ========================================")
            logger.info(f"[TOOL_SOURCE] Gym #{idx + 1}/{len(self.gym_configs)}")
            logger.info(f"[TOOL_SOURCE] Server Name: {gym_name}")
            logger.info(f"[TOOL_SOURCE] Server URL: {gym_config['mcp_server_url']}")
            logger.info(f"[TOOL_SOURCE] Database ID: {gym_config['database_id']}")
            logger.info(f"[TOOL_SOURCE] ========================================")

            try:
                server_tools = await client.list_tools()
                tool_names = [t.get("name", "unknown") for t in server_tools]

                logger.info(
                    f"[TOOL_SOURCE] ✅ Discovered {len(server_tools)} tools from gym '{gym_name}'"
                )
                logger.info(
                    f"[TOOL_SOURCE] Tool names: {', '.join(tool_names[:10])}{' ...' if len(tool_names) > 10 else ''}"
                )

                # Add server metadata to each tool and merge
                for tool in server_tools:
                    tool_name = tool.get("name", "unknown")

                    # Check for tool name conflicts across gyms
                    if tool_name in tool_to_server_mapping:
                        existing_gym = tool_to_server_mapping[tool_name]
                        logger.warning(
                            f"[TOOL_SOURCE] ⚠️  Tool '{tool_name}' DUPLICATE! "
                            f"Found in '{existing_gym}' and '{gym_name}'. "
                            f"Using first occurrence from '{existing_gym}'."
                        )
                        continue  # Skip duplicate

                    # Enhance tool with gym metadata
                    enhanced_tool = tool.copy()
                    enhanced_tool["_mcp_server_name"] = gym_name
                    enhanced_tool["_mcp_server_url"] = gym_config["mcp_server_url"]
                    enhanced_tool["_database_id"] = gym_config["database_id"]

                    merged_tools.append(enhanced_tool)
                    tool_to_server_mapping[tool_name] = gym_name

                    logger.debug(
                        f"[TOOL_SOURCE] Tool '{tool_name}' → Gym: {gym_name}, DB: {gym_config['database_id']}"
                    )

            except Exception as e:
                logger.error(f"Failed to discover tools from gym {gym_name}: {e}")
                raise

        # Filter tools based on selected_tools if configured
        if self.config.selected_tools and len(self.config.selected_tools) > 0:
            logger.info(
                f"[TOOL_FILTER] 🔧 Filtering tools based on selected_tools list ({len(self.config.selected_tools)} tools)"
            )
            logger.info(
                f"[TOOL_FILTER] Expected tools: {', '.join(self.config.selected_tools)}"
            )
            filtered_tools = [
                tool
                for tool in merged_tools
                if tool.get("name") in self.config.selected_tools
            ]
            logger.info(
                f"[TOOL_FILTER] ✅ Filtered from {len(merged_tools)} to {len(filtered_tools)} tools"
            )
            if len(filtered_tools) < len(self.config.selected_tools):
                missing = set(self.config.selected_tools) - {
                    t.get("name") for t in filtered_tools
                }
                logger.warning(f"[TOOL_FILTER] ⚠️  Expected tools not found: {missing}")
            self.available_tools = filtered_tools
        else:
            logger.info(
                f"[TOOL_FILTER] ℹ️  No tool filtering applied - using all {len(merged_tools)} discovered tools"
            )
            logger.info(
                f"[TOOL_FILTER] ℹ️  To limit tools and reduce token usage, configure 'selected_tools' in your config"
            )
            self.available_tools = merged_tools

        self.tool_to_server_mapping = tool_to_server_mapping

    async def execute_single_run(self, run_number: int) -> Dict[str, Any]:
        """Execute a single benchmark run"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING RUN {run_number}/{self.config.number_of_runs}")
        logger.info(f"{'='*80}\n")

        start_time = datetime.now(timezone.utc)

        # Track databases created in this run
        run_databases = []

        for gym_conf in self.gym_configs:
            logger.info(f"===============================================")
            logger.info(f"Preparing gym: {gym_conf}")
            db_id = create_database_from_file(
                gym_conf["mcp_server_url"], gym_conf["seed_database_file"]
            )

            if db_id:
                # Track this database for cleanup
                run_databases.append(
                    {
                        "gym_name": gym_conf["mcp_server_name"],
                        "gym_url": gym_conf["mcp_server_url"],
                        "database_id": db_id,
                    }
                )
                self.auto_created_databases.append(
                    {
                        "gym_name": gym_conf["mcp_server_name"],
                        "gym_url": gym_conf["mcp_server_url"],
                        "database_id": db_id,
                    }
                )

            client = self.mcp_clients.get(gym_conf["mcp_server_name"])
            print(
                f"Created database ID: {db_id} for gym: {gym_conf['mcp_server_name']}"
            )
            if client:
                client.database_id = db_id

        # Execute the main task via the configured orchestrator
        orchestrator = self.orchestrator_class(
            llm_client=self.llm_client,
            mcp_clients=self.mcp_clients,
            tool_to_server_mapping=self.tool_to_server_mapping,
            available_tools=self.available_tools,
            config=self.config,
            **self.orchestrator_kwargs,
        )
        task_result = await orchestrator.execute()

        # Run verifiers
        verification_results = await self._run_verifiers(task_result)

        # Calculate execution time
        execution_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        # Determine overall success
        overall_success = all(v["passed"] for v in verification_results.values())

        # Calculate verification summary
        total_verifiers = len(verification_results)
        passed_verifiers = sum(
            1 for v in verification_results.values() if v.get("passed", False)
        )
        failed_verifiers = total_verifiers - passed_verifiers

        result = {
            "run_number": run_number,
            "started_at": start_time.isoformat(),
            "execution_time_ms": execution_time_ms,
            "model_response": task_result.get("final_response"),
            "conversation_flow": task_result.get("conversation_flow", []),
            "tools_used": task_result.get("tools_used", []),
            "tool_results": task_result.get("tool_results", []),
            "verification_results": verification_results,
            "verification_summary": {
                "total": total_verifiers,
                "passed": passed_verifiers,
                "failed": failed_verifiers,
                "pass_rate": (
                    passed_verifiers / total_verifiers if total_verifiers > 0 else 0.0
                ),
            },
            "overall_success": overall_success,
        }

        result.update(orchestrator.get_result_metadata())

        logger.info(f"\nRUN {run_number} COMPLETED")
        logger.info(
            f"Verification: {passed_verifiers}/{total_verifiers} passed ({passed_verifiers/total_verifiers*100:.1f}%)"
        )
        logger.info(f"Overall Success: {overall_success}")
        logger.info(f"Execution time: {execution_time_ms}ms")
        logger.info(f"Tools used: {', '.join(task_result.get('tools_used', []))}")

        return result

    async def _run_verifiers(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured verifiers"""
        logger.info("\n--- Running Verifiers ---")

        verification_results = {}

        for i, verifier_config in enumerate(self.config.verifiers):
            verifier = VerifierConfig(**verifier_config)
            verifier_name = verifier.name or f"verifier_{i+1}"

            # logger.info(f"Running verifier: {verifier_name} ({verifier.verifier_type})")

            # Determine which database_id and context to use
            database_id = None
            context = None

            if verifier.gym_name:
                # Multi-gym: Find the gym config by name
                gym_config = next(
                    (
                        g
                        for g in self.gym_configs
                        if g["mcp_server_name"] == verifier.gym_name
                    ),
                    None,
                )
                if gym_config:
                    database_id = gym_config.get("database_id")
                    context = gym_config.get("context", {})
                    # logger.info(f"  → Using gym '{verifier.gym_name}' database: {database_id}")
                else:
                    logger.warning(
                        f"  ⚠️ Gym '{verifier.gym_name}' not found in gym_servers_config!"
                    )
                    # verification_results[verifier_name] = {
                    #     "passed": False,
                    #     "error": f"Gym '{verifier.gym_name}' not found in configuration",
                    # }
                    continue
            else:
                # Single-gym or legacy: Use default database_id
                database_id = self.config.database_id
                context = self.config.context
                # if len(self.gym_configs) > 1:
                #     logger.warning(f"  ⚠️ Multi-gym setup but verifier '{verifier_name}' has no gym_name specified!")

            model_response = {
                "content": task_result.get("final_response", ""),
                "tool_calls": [
                    {"name": tr["tool_name"], "args": tr["arguments"]}
                    for tr in task_result.get("tool_results", [])
                ],
            }

            result = await self.verifier_engine.execute_verifier(
                verifier,
                model_response,
                database_id,
                context,
                gym_name=verifier.gym_name,  # Pass gym_name to use correct MCP client
            )

            verification_results[verifier_name] = result

            # logger.info(f"Verifier result: {'✓ PASSED' if result.get('passed') else '✗ FAILED'}")
            # if not result.get("passed"):
            #     logger.warning(f"Failure reason: {result.get('error') or result.get('details')}")

        return verification_results

    async def execute_benchmark(self) -> Dict[str, Any]:
        """Execute complete benchmark with multiple runs"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING BENCHMARK EXECUTION")
        logger.info(
            f"Model: {self.llm_config.llm_provider}/{self.llm_config.llm_model}"
        )
        logger.info(f"Number of runs: {self.config.number_of_runs}")
        logger.info(f"{'='*80}\n")

        # Parse gym configurations first
        self.gym_configs = self._parse_gym_configs()

        # Get the directory containing the config file (for resolving relative paths)
        config_dir = os.path.dirname(os.path.abspath(self.config_path))

        # Create master databases at the start of benchmark
        logger.info(f"\n{'='*80}")
        logger.info("DATABASE SETUP - Creating master databases for all gyms")
        logger.info(f"{'='*80}\n")

        logger.info(f"\n✅ All master databases created successfully\n")

        try:
            await self.initialize()

            all_runs = []

            for run_number in range(1, self.config.number_of_runs + 1):
                try:
                    run_result = await self.execute_single_run(run_number)
                    all_runs.append(run_result)
                except Exception as e:
                    logger.error(f"Run {run_number} failed with error: {e}")
                    all_runs.append(
                        {
                            "run_number": run_number,
                            "error": str(e),
                            "overall_success": False,
                        }
                    )

            # Calculate statistics
            statistics = self._calculate_statistics(all_runs)
            result = {
                "benchmark_config": {
                    "model": f"{self.llm_config.llm_provider}/{self.llm_config.llm_model}",
                    "number_of_runs": self.config.number_of_runs,
                    "user_prompt": self.config.user_prompt,
                    "gym_servers": [
                        {
                            "name": g["mcp_server_name"],
                            "url": g["mcp_server_url"],
                            "seed_database_file": g.get("seed_database_file", ""),
                            "uses_cloning": True,
                        }
                        for g in self.gym_configs
                    ],
                    "total_gyms": len(self.gym_configs),
                    "total_tools_available": len(self.available_tools),
                },
                "runs": all_runs,
                "statistics": statistics,
            }

            logger.info(f"\n{'='*80}")
            logger.info(f"BENCHMARK COMPLETED")
            logger.info(f"{'='*80}\n")

            return result

        finally:
            # Cleanup: Delete auto-created databases
            if self.auto_created_databases:
                logger.info(
                    f"\n🧹 Cleaning up {len(self.auto_created_databases)} auto-created database(s)..."
                )
                for db_info in self.auto_created_databases:
                    logger.info(
                        f"Deleting database for gym '{db_info['gym_name']}': {db_info['database_id']}"
                    )
                    delete_database(db_info["gym_url"], db_info["database_id"])

    def _calculate_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate benchmark statistics"""
        successful_runs = [r for r in runs if r.get("overall_success")]
        total_runs = len(runs)

        # Overall success rate (all verifiers must pass)
        overall_success_rate = (
            len(successful_runs) / total_runs if total_runs > 0 else 0
        )

        # Pass@1: success on first run
        pass_at_1 = 1.0 if runs and runs[0].get("overall_success") else 0.0

        # Verifier-level statistics
        total_verifiers_count = 0
        passed_verifiers_count = 0
        verifier_pass_rates = {}

        for run in runs:
            if "verification_summary" in run:
                total_verifiers_count += run["verification_summary"]["total"]
                passed_verifiers_count += run["verification_summary"]["passed"]

            # Track individual verifier pass rates
            for verifier_name, result in run.get("verification_results", {}).items():
                if verifier_name not in verifier_pass_rates:
                    verifier_pass_rates[verifier_name] = {"passed": 0, "total": 0}
                verifier_pass_rates[verifier_name]["total"] += 1
                if result.get("passed", False):
                    verifier_pass_rates[verifier_name]["passed"] += 1

        # Calculate pass rate for each verifier
        verifier_stats = {}
        for verifier_name, counts in verifier_pass_rates.items():
            verifier_stats[verifier_name] = {
                "passed": counts["passed"],
                "total": counts["total"],
                "pass_rate": (
                    counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0
                ),
            }

        # Overall verifier pass rate
        verifier_level_pass_rate = (
            passed_verifiers_count / total_verifiers_count
            if total_verifiers_count > 0
            else 0
        )

        # Mean execution time
        execution_times = [
            r.get("execution_time_ms", 0) for r in runs if "execution_time_ms" in r
        ]
        mean_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )

        # Tool usage
        all_tools = []
        for run in runs:
            all_tools.extend(run.get("tools_used", []))

        tool_counts = {}
        for tool in all_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_runs": total_runs,
            "successful_runs": len(successful_runs),
            "overall_success_rate": overall_success_rate,
            "pass_at_1": pass_at_1,
            "verifier_level_pass_rate": verifier_level_pass_rate,
            "total_verifiers_checked": total_verifiers_count,
            "total_verifiers_passed": passed_verifiers_count,
            "individual_verifier_stats": verifier_stats,
            "mean_execution_time_ms": mean_time,
            "tool_usage": tool_counts,
        }
