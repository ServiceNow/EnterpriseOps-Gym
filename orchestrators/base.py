"""Base class for agent orchestrators."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from benchmark.mcp_client import MCPClient
from benchmark.llm_client import LLMClient
from benchmark.models import BenchmarkConfig

logger = logging.getLogger(__name__)


class AgentOrchestrator(ABC):

    # True only for orchestrators that accumulate their trace on self as they
    # go. Where it is False, an empty partial trace means "not captured", not
    # "the model did nothing" — consumers must be able to tell those apart.
    supports_partial_trace = False

    def __init__(
        self,
        llm_client: "LLMClient",
        mcp_clients: Dict[str, "MCPClient"],
        tool_to_server_mapping: Dict[str, str],
        available_tools: List[Dict[str, Any]],
        config: "BenchmarkConfig",
        max_iterations: int = 50,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        self.llm_client = llm_client
        self.mcp_clients = mcp_clients
        self.tool_to_server_mapping = tool_to_server_mapping
        # Exposed (prefixed) name -> the name the MCP server knows. Tools that
        # are not prefixed are absent and keep their own name.
        self.tool_name_mapping = tool_name_mapping or {}
        self.available_tools = available_tools
        self.config = config
        self.max_iterations = max_iterations
        # Accumulated on self (not in locals) so a run that raises mid-loop
        # still yields the trace collected up to the failure.
        self.conversation_flow: List[Dict[str, Any]] = []
        self.tools_used: List[str] = []
        self.tool_results: List[Dict[str, Any]] = []

    def get_partial_trace(self) -> Dict[str, Any]:
        """Trace collected so far. Used to preserve evidence when a run fails
        before execute() returns."""
        return {
            "conversation_flow": self.conversation_flow,
            "tools_used": self.tools_used,
            "tool_results": self.tool_results,
            "partial_trace_captured": self.supports_partial_trace,
        }

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
            # A model can call a tool it was never given. Report it back as a
            # failed call so the trace explains the absent mutation, instead of
            # raising and discarding the run.
            logger.error(
                f"Tool '{tool_name}' is not in the tool-to-server mapping; "
                f"reporting call as failed."
            )
            return {
                "result": {
                    "success": False,
                    "error": f"Unknown tool '{tool_name}': not exposed to this task.",
                },
                "gym_server": None,
            }

        client = self.mcp_clients[target_gym]
        logger.info(f"Executing '{tool_name}' on '{target_gym}'")

        result = await client.call_tool(
            self.tool_name_mapping.get(tool_name, tool_name), tool_args
        )

        return {"result": result, "gym_server": target_gym}