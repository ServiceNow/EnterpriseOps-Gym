"""Live environment for one human session, built entirely from upstream pieces.

Lifecycle mirrors `BenchmarkExecutor.execute_single_run`:
    seed a fresh database per gym -> connect MCP clients -> discover/filter tools
    -> (human makes tool calls) -> run upstream verifiers -> delete databases.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from benchmark.executor import BenchmarkExecutor
from benchmark.mcp_client import MCPClient, create_database_from_file, delete_database
from benchmark.verifier import VerifierEngine
from evaluate import load_config

from . import REPO_ROOT
from .provenance import docker_image_for_port, sha256_file

logger = logging.getLogger(__name__)


def _resolve_seed(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


class TaskEnvironment:
    def __init__(self, task_config: Dict[str, Any]):
        fd, path = tempfile.mkstemp(suffix=".json", prefix="datasea_task_")
        with os.fdopen(fd, "w") as f:
            json.dump(task_config, f)
        try:
            self.config = load_config(path)
        finally:
            os.unlink(path)
        self.executor = BenchmarkExecutor(self.config, llm_config=None)
        self.executor.gym_configs = self.executor._parse_gym_configs()
        self.tools: List[Dict[str, Any]] = []
        self.environment_info: List[Dict[str, Any]] = []

    async def start(self) -> None:
        ex = self.executor
        for gym in ex.gym_configs:
            seed = _resolve_seed(gym["seed_database_file"])
            db_id = await asyncio.to_thread(create_database_from_file, gym["mcp_server_url"], seed)
            gym["database_id"] = db_id
            client = MCPClient(
                base_url=gym["mcp_server_url"],
                auth_config=gym.get("auth_config"),
                mcp_endpoint=gym.get("mcp_endpoint", "/mcp"),
                database_id=db_id,
                context=gym.get("context", {}),
            )
            if not await client.connect():
                raise RuntimeError(f"Failed to connect to MCP server {gym['mcp_server_name']} at {gym['mcp_server_url']}")
            ex.mcp_clients[gym["mcp_server_name"]] = client
            port = int(gym["mcp_server_url"].rsplit(":", 1)[-1].split("/")[0])
            self.environment_info.append(
                {
                    "gym_name": gym["mcp_server_name"],
                    "mcp_server_url": gym["mcp_server_url"],
                    "database_id": db_id,
                    "seed_database_file": gym["seed_database_file"],
                    "seed_sha256": sha256_file(seed),
                    **docker_image_for_port(port),
                }
            )

        await ex._discover_and_merge_tools()
        if self.config.restricted_tools:
            ex.available_tools = [t for t in ex.available_tools if t["name"] not in self.config.restricted_tools]
        self.tools = [
            {"name": t["name"], "description": t.get("description", ""), "inputSchema": t.get("inputSchema", {})}
            for t in ex.available_tools
        ]
        ex.verifier_engine = VerifierEngine(ex.mcp_clients, llm_client=None)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Same routing as AgentOrchestrator._execute_tool_call."""
        ex = self.executor
        if tool_name not in {t["name"] for t in self.tools}:
            return {"success": False, "error": f"Tool '{tool_name}' is not available for this task"}
        return await ex.mcp_clients[ex.tool_to_server_mapping[tool_name]].call_tool(tool_name, arguments)

    async def verify(self, final_response: str, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_result = {"final_response": final_response, "tool_results": tool_results}
        results = await self.executor._run_verifiers(task_result)
        passed = sum(1 for v in results.values() if v.get("passed"))
        return {
            "overall_success": bool(results) and passed == len(results),
            "verification_results": results,
            "verification_summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        }

    async def reset(self) -> None:
        """Drop this session's databases; the next session seeds fresh ones from the snapshot."""
        for gym in self.executor.gym_configs:
            if gym.get("database_id"):
                await asyncio.to_thread(delete_database, gym["mcp_server_url"], gym["database_id"])
                gym["database_id"] = ""
