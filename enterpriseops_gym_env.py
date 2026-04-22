"""EnterpriseOps-Gym adapter for the Prime Intellect Verifiers framework.

Wraps ServiceNow's EnterpriseOps-Gym benchmark (1,150 enterprise agent tasks across
8 domains, verified via SQL state checks) as a ``vf.ToolEnv`` so it can be hosted on
Prime Intellect's Environment Hub.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any, cast

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.types import Messages, State, Tool, ToolMessage

from benchmark.llm_client import LLMClient
from benchmark.mcp_client import MCPClient, create_database_from_file, delete_database
from benchmark.models import VerifierConfig
from benchmark.verifier import VerifierEngine

logger = logging.getLogger(__name__)

# MCP server names (as referenced in the HuggingFace dataset's gym_servers_config)
# mapped to the default localhost ports from EnterpriseOps-Gym's Docker setup.
# IMPORTANT: Most containers listen internally on port 8005 (calendar on 8003).
# Use e.g. `docker run -p 8002:8005 ...teams:latest` — see README for full commands.
DEFAULT_SERVER_URLS: dict[str, str] = {
    "sn-csm-server": "http://localhost:8001",
    "gym-teams-mcp": "http://localhost:8002",
    "gym-calendar": "http://localhost:8003",
    "gym-email-mcp": "http://localhost:8004",
    "gym-itsm-mcp": "http://localhost:8006",
    "sn-hr-internal": "http://localhost:8008",
    "gym-google-drive-mcp": "http://localhost:8009",
}

ALL_DOMAINS = ["teams", "csm", "calendar", "email", "itsm", "hr", "drive", "hybrid"]


# -- Helpers ------------------------------------------------------------------


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from a synchronous context.

    Uses ``asyncio.run()`` when no event loop is running, otherwise falls back
    to executing in a thread (e.g. inside Jupyter).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()



def _parse_info(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Deserialize the info field, which is stored as a JSON string in the dataset."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _resolve_sql_path(seed_database_file: str, gym_dbs_path: str) -> str | None:
    """Try several candidate locations for a SQL seed file."""
    candidates = [
        seed_database_file,
        os.path.join(gym_dbs_path, seed_database_file),
        os.path.join(gym_dbs_path, os.path.basename(seed_database_file)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _mcp_content_to_str(result: Any) -> str:
    """Flatten an MCP tool result into a plain string for the conversation."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [
                item["text"] if isinstance(item, dict) and item.get("type") == "text" else str(item)
                for item in content
            ]
            if parts:
                return "\n".join(parts)
        return str(content) if content else json.dumps(result)
    return str(result)


# -- Environment --------------------------------------------------------------


class EOpsGymEnv(vf.ToolEnv):
    """EnterpriseOps-Gym as a Verifiers ``ToolEnv``.

    Connects to pre-running MCP Docker servers at init, discovers all tools,
    then per rollout: seeds fresh databases, scopes the tool set to the task's
    ``selected_tools``, runs the agent loop, scores via the benchmark's
    ``VerifierEngine``, and cleans up databases.
    """

    def __init__(
        self,
        server_urls: dict[str, str],
        gym_dbs_path: str,
        max_turns: int = 50,
        llm_client: LLMClient | None = None,
        **kwargs: Any,
    ):
        self.server_urls = server_urls
        self.gym_dbs_path = gym_dbs_path
        self.llm_client = llm_client
        self.clients: dict[str, MCPClient] = {}
        self.tool_to_server: dict[str, str] = {}
        self._all_tool_defs: list[Tool] = []

        self._connect_and_discover()

        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        self.tool_defs = list(self._all_tool_defs)

    # -- Init helpers ---------------------------------------------------------

    def _connect_and_discover(self) -> None:
        """Connect to every configured MCP server and merge their tool catalogues."""
        for name, url in self.server_urls.items():
            client = MCPClient(base_url=url)
            if not _run_sync(client.connect()):
                logger.warning("Could not connect to %s at %s — skipping", name, url)
                continue
            self.clients[name] = client

            for schema in _run_sync(client.list_tools()):
                tool_name = schema["name"]
                if tool_name in self.tool_to_server:
                    logger.debug("Duplicate tool '%s' — keeping first occurrence", tool_name)
                    continue
                self.tool_to_server[tool_name] = name
                self._all_tool_defs.append(
                    Tool(
                        name=tool_name,
                        description=schema.get("description", ""),
                        parameters=schema.get("inputSchema", {"type": "object", "properties": {}}),
                    )
                )

        logger.info(
            "EOpsGymEnv: discovered %d tools across %d servers", len(self._all_tool_defs), len(self.clients)
        )

    # -- Per-rollout lifecycle ------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Seed databases and scope tools for the current task."""
        state = await super().setup_state(state)

        # Retrieve per-task metadata from the dataset row
        info = _parse_info(state["input"]["info"])

        gym_configs: list[dict] = info.get("gym_servers_config", [])
        selected: list[str] = info.get("selected_tools", [])

        # Seed a database for each MCP server this task uses
        db_ids: dict[str, str] = {}
        url_map: dict[str, str] = {}

        for cfg in gym_configs:
            srv = cfg["mcp_server_name"]
            url = self.server_urls.get(srv, cfg.get("mcp_server_url", ""))
            url_map[srv] = url

            seed_file = cfg.get("seed_database_file", "")
            if not seed_file:
                continue

            sql_path = _resolve_sql_path(seed_file, self.gym_dbs_path)
            if not sql_path:
                logger.warning("SQL seed file not found for %s: %s", srv, seed_file)
                continue

            db_id = create_database_from_file(url, sql_path)
            if not db_id:
                logger.warning("Failed to create database for %s", srv)
                continue
            db_ids[srv] = db_id
            if srv in self.clients:
                self.clients[srv].database_id = db_id

        state["database_ids"] = db_ids
        state["server_url_map"] = url_map

        # Restrict visible tools to this task's selected set
        if selected:
            state["tool_defs"] = [t for t in self._all_tool_defs if t.name in selected]
        else:
            state["tool_defs"] = list(self._all_tool_defs)

        return state

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs: Any
    ) -> ToolMessage:
        """Route a tool call to the owning MCP server via HTTP JSON-RPC."""
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            return cast(
                ToolMessage,
                {"role": "tool", "content": f"Error: unknown tool '{tool_name}'", "tool_call_id": tool_call_id},
            )

        client = self.clients.get(server_name)
        if not client:
            return cast(
                ToolMessage,
                {"role": "tool", "content": f"Error: server '{server_name}' not connected", "tool_call_id": tool_call_id},
            )

        result = await client.call_tool(tool_name, tool_args)
        if result.get("success"):
            content = _mcp_content_to_str(result.get("result", ""))
        else:
            content = f"Error: {result.get('error', 'unknown')}"

        return cast(ToolMessage, {"role": "tool", "content": content, "tool_call_id": tool_call_id})

    # NOTE: Database cleanup is intentionally NOT done via @vf.cleanup because that hook
    # runs inside rollout(), BEFORE the rubric scores the state. The rubric needs the
    # databases alive to run SQL verifiers. Instead, cleanup is done via rubric.cleanup()
    # which runs AFTER scoring. See _build_rubric().

    @vf.teardown
    async def _teardown(self) -> None:
        """Release MCP clients on environment shutdown."""
        self.clients.clear()


# -- Dataset builder ----------------------------------------------------------


def _build_dataset(hf_repo: str, mode: str, domains: list[str]) -> Dataset:
    """Load the HuggingFace dataset and reshape into Verifiers format.

    Each row becomes:
        prompt:  [{role: system, ...}, {role: user, ...}]
        answer:  "" (verification is SQL-based, not text-based)
        info:    {task_id, domain, selected_tools, verifiers, gym_servers_config, ...}
    """
    json_fields = {"gym_servers_config", "verifiers"}
    rows: list[dict[str, Any]] = []

    for domain in domains:
        logger.info("Loading HF dataset: %s config=%s split=%s", hf_repo, mode, domain)
        ds = load_dataset(hf_repo, mode, split=domain)
        for raw in ds:
            info: dict[str, Any] = {}
            for k, v in raw.items():
                if k in ("system_prompt", "user_prompt"):
                    continue
                if k in json_fields and isinstance(v, str):
                    info[k] = json.loads(v)
                else:
                    info[k] = v

            rows.append({
                "prompt": [
                    {"role": "system", "content": raw["system_prompt"]},
                    {"role": "user", "content": raw["user_prompt"]},
                ],
                "answer": "",
                "info": json.dumps(info),  # serialized to avoid Arrow type-inference conflicts
            })

    logger.info("Dataset built: %d tasks across %s", len(rows), domains)
    return Dataset.from_dict({
        "prompt": [r["prompt"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "info": [r["info"] for r in rows],
    })


# -- Rubric builder -----------------------------------------------------------


def _collect_tool_calls(state: State) -> list[str]:
    """Extract the list of tool names called during the rollout from the trajectory."""
    tool_names: list[str] = []
    for step in state["trajectory"]:
        for msg in step["completion"]:
            if msg.role != "assistant":
                continue
            for tc in msg.tool_calls or []:
                if tc.name:
                    tool_names.append(tc.name)
    return tool_names


def _build_rubric(server_urls: dict[str, str], llm_client: LLMClient | None = None) -> vf.Rubric:
    """Create a rubric that scores rollouts using the benchmark's ``VerifierEngine``.

    Delegates all verification logic (database_state, tool_execution, response_check)
    to ``benchmark.verifier.VerifierEngine``, avoiding duplicated verification code.

    Args:
        server_urls: MCP server name to base URL mapping (for constructing per-verifier clients).
        llm_client: Optional ``benchmark.llm_client.LLMClient`` for ``response_check`` verifiers.
    """

    async def verification(completion: Messages, answer: str, state: State, info: str) -> float:
        """Fraction of verifiers that pass, delegated to VerifierEngine."""
        info = _parse_info(info)
        verifier_configs: list[dict] = info.get("verifiers", [])
        if not verifier_configs:
            return 0.0

        db_ids: dict[str, str] = state.get("database_ids", {})
        url_map: dict[str, str] = state.get("server_url_map", {})

        # Build MCP clients dict for VerifierEngine (keyed by gym_name)
        mcp_clients: dict[str, MCPClient] = {}
        for gym_name in {v.get("gym_name") for v in verifier_configs if v.get("gym_name")}:
            base_url = url_map.get(gym_name, server_urls.get(gym_name, ""))
            db_id = db_ids.get(gym_name, "")
            if base_url:
                mcp_clients[gym_name] = MCPClient(base_url=base_url, database_id=db_id)

        # Skip response_check verifiers if no llm_client is configured
        runnable = []
        for v_cfg in verifier_configs:
            if v_cfg.get("verifier_type") == "response_check" and llm_client is None:
                logger.warning(
                    "Skipping response_check verifier (no llm_client configured). "
                    "Pass llm_client to load_environment() to enable."
                )
                continue
            runnable.append(v_cfg)

        if not runnable:
            return 0.0

        engine = VerifierEngine(mcp_clients, llm_client)

        # Build model_response for tool_execution and response_check verifiers
        tools_called = _collect_tool_calls(state)
        model_response = {
            "content": completion and completion[-1].content or "",
            "tool_calls": [{"name": t, "args": {}} for t in tools_called],
        }

        passed = 0
        total = len(runnable)

        for v_cfg in runnable:
            verifier = VerifierConfig(**v_cfg)
            db_id = db_ids.get(verifier.gym_name, "")
            try:
                result = await engine.execute_verifier(verifier, model_response, db_id, gym_name=verifier.gym_name)
                if result.get("passed"):
                    passed += 1
            except Exception:
                logger.debug("Verifier failed", exc_info=True)

        return passed / total

    async def all_pass(completion: Messages, answer: str, state: State, info: str) -> float:
        """Binary metric: 1.0 only if every verifier passes."""
        info = _parse_info(info)
        if not info.get("verifiers"):
            return 0.0
        score = await verification(completion, answer, state, info)
        return 1.0 if score == 1.0 else 0.0

    async def cleanup_databases(state: State) -> None:
        """Delete databases created for this rollout. Runs after scoring."""
        url_map: dict[str, str] = state.get("server_url_map", {})
        for srv, db_id in state.get("database_ids", {}).items():
            url = url_map.get(srv, server_urls.get(srv, ""))
            if url:
                delete_database(url, db_id)

    rubric = vf.Rubric(funcs=[verification], weights=[1.0])
    rubric.add_metric(all_pass)
    rubric._cleanup_handlers.append(cleanup_databases)
    return rubric


# -- Entry point --------------------------------------------------------------


def load_environment(
    server_urls: dict[str, str] | None = None,
    gym_dbs_path: str = "gym_dbs",
    hf_dataset: str = "ServiceNow-AI/EnterpriseOps-Gym",
    mode: str = "oracle",
    domains: list[str] | None = None,
    max_turns: int = 50,
    llm_client: LLMClient | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load EnterpriseOps-Gym as a Verifiers environment.

    Args:
        server_urls: MCP server name to base URL mapping.
            Defaults to localhost with the standard ports from the EnterpriseOps-Gym Docker setup.
        gym_dbs_path: Path to directory with extracted SQL seed files (from ``gym_dbs.zip``).
        hf_dataset: HuggingFace dataset repo ID.
        mode: Tool-set mode (``oracle``, ``plus_5_tools``, ``plus_10_tools``, ``plus_15_tools``).
        domains: Which domains to include. Defaults to all 8 (7 single-domain + hybrid).
        max_turns: Maximum agent turns per task.
        llm_client: Optional ``benchmark.llm_client.LLMClient`` instance for ``response_check``
            verifiers. If not provided, ``response_check`` verifiers will fail gracefully.

    Returns:
        A configured ``EOpsGymEnv`` ready for evaluation.
    """
    urls = server_urls or dict(DEFAULT_SERVER_URLS)
    doms = domains or list(ALL_DOMAINS)

    dataset = _build_dataset(hf_dataset, mode, doms)
    rubric = _build_rubric(urls, llm_client)

    return EOpsGymEnv(
        server_urls=urls,
        gym_dbs_path=gym_dbs_path,
        max_turns=max_turns,
        llm_client=llm_client,
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )
