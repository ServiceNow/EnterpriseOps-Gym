"""Base class for agent orchestrators."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

from benchmark.mcp_client import MCPClient
from benchmark.llm_client import LLMClient
from benchmark.models import BenchmarkConfig

logger = logging.getLogger(__name__)


class AgentOrchestrator(ABC):

    def __init__(
        self,
        llm_client: "LLMClient",
        mcp_clients: Dict[str, "MCPClient"],
        tool_to_server_mapping: Dict[str, str],
        available_tools: List[Dict[str, Any]],
        config: "BenchmarkConfig",
        max_iterations: int = 50,
    ):
        self.llm_client = llm_client
        self.mcp_clients = mcp_clients
        self.tool_to_server_mapping = tool_to_server_mapping
        self.available_tools = available_tools
        self.config = config
        self.max_iterations = max_iterations

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the task and return results dict with keys:
        final_response, conversation_flow, tools_used, tool_results, messages
        """
        pass

    def get_result_metadata(self) -> Dict[str, Any]:
        """Return extra metadata to merge into the run result.
        Override in subclasses to surface orchestrator-specific telemetry
        (e.g. token usage, plan metadata).
        """
        return {}

    async def _execute_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        target_gym = self.tool_to_server_mapping.get(tool_name)

        if not target_gym:
            target_gym = list(self.mcp_clients.keys())[0]
            logger.error(f"Tool '{tool_name}' not in mapping, using: {target_gym}")
            raise

        client = self.mcp_clients[target_gym]
        logger.info(f"Executing '{tool_name}' on '{target_gym}'")

        result = await client.call_tool(tool_name, tool_args)

        return {"result": result, "gym_server": target_gym}