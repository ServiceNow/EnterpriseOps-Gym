"""Golden orchestrator - replays the manual (oracle) tool executions.

This orchestrator involves no LLM. Instead of asking a model what to do next,
it reads the golden solution (``manual_tool_executions``) from the raw task file
and replays each tool call, with the exact same arguments, live against the
seeded database - conceptually like ReAct, but the "reasoning" is the fixed
golden trajectory.

The raw task file is located from ``config_path`` using a static path template
(see ``RAW_DATA_ROOT`` / ``RAW_PATH_TEMPLATE`` below). The golden actions live
at::

    golden_task['executions']['Oracle']['config']['manual_tool_executions']

Each entry looks like::

    {
        "tool_name": "search_cases",
        "server_name": "sn-csm-server",   # often null - not used for routing
        "arguments": {"number": "CS-0000058"},
        "results": [ ... ]                # cached golden result (logging only)
    }

Routing is done through the executor's ``tool_to_server_mapping`` (via the
inherited ``_execute_tool_call``), not the stored ``server_name`` (which is
frequently ``None``), so single-gym and multi-gym (hybrid) tasks both work.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from .base import AgentOrchestrator
from .golden_common import GoldenFileMixin, GOLDEN_FINAL_RESPONSE

logger = logging.getLogger(__name__)


class GoldenOrchestrator(GoldenFileMixin, AgentOrchestrator):
    """Replays the golden ``manual_tool_executions`` against the live DB."""

    def __init__(self, *args, config_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def execute(self) -> Dict[str, Any]:
        """Replay the golden tool sequence step by step against the DB."""
        manual_tool_executions = self._load_manual_tool_executions()

        # Minimal message trail so downstream serialization has the expected
        # shape (no LLM is involved, so there are no AI messages).
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self.config.user_prompt),
        ]

        conversation_flow = [
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
        ]
        tools_used: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for idx, step in enumerate(manual_tool_executions):
            tool_name = step.get("tool_name")
            tool_args = step.get("arguments") or {}

            if not tool_name:
                logger.warning(
                    f"[GOLDEN] Step {idx + 1} has no tool_name; skipping: {step}"
                )
                continue

            logger.info(
                f"\n--- Golden Step {idx + 1}/{len(manual_tool_executions)}: "
                f"{tool_name} ---"
            )
            logger.debug(f"[GOLDEN] Arguments: {tool_args}")

            # Execute live against the seeded DB. Routing uses the executor's
            # tool_to_server_mapping (not the stored server_name).
            exec_result = await self._execute_tool_call(tool_name, tool_args)
            tool_result = exec_result["result"]
            target_gym = exec_result["gym_server"]

            logger.info(f"[GOLDEN] Tool result success: {tool_result.get('success')}")

            if tool_name not in tools_used:
                tools_used.append(tool_name)

            tool_results.append(
                {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                    "gym_server": target_gym,
                }
            )

            conversation_flow.append(
                {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": tool_result,
                    "gym_server": target_gym,
                }
            )

        return {
            "final_response": GOLDEN_FINAL_RESPONSE,
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }
