"""ReAct orchestrator - reasoning and acting in a loop."""

import json
import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

from benchmark.mcp_client import MCPClient
from benchmark.llm_client import LLMClient, get_text_content
from benchmark.models import BenchmarkConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from .base import AgentOrchestrator

logger = logging.getLogger(__name__)


def apply_prompt_overlay(system_prompt: str, user_prompt: str):
    """Wrap task prompts with meta-instructions from the JSON file named by $PROMPT_OVERLAY.

    Keys: system_prefix, system_suffix, user_prefix, user_suffix, policy_to_user (bool).
    The task-supplied text is always passed through verbatim.
    """
    path = os.environ.get("PROMPT_OVERLAY")
    if not path:
        return system_prompt, user_prompt
    with open(path) as f:
        ov = json.load(f)
    sp, ss = ov.get("system_prefix", ""), ov.get("system_suffix", "")
    up, us = ov.get("user_prefix", ""), ov.get("user_suffix", "")
    if ov.get("policy_to_user"):
        new_system = (sp + ss).strip() or "You are a helpful enterprise operations agent."
        new_user = (
            f"<operating_policy>\n{system_prompt}\n</operating_policy>\n\n"
            f"{up}<user_request>\n{user_prompt}\n</user_request>{us}"
        )
        return new_system, new_user
    return f"{sp}{system_prompt}{ss}", f"{up}{user_prompt}{us}"


class ReactOrchestrator(AgentOrchestrator):

    supports_partial_trace = True

    async def execute(self) -> Dict[str, Any]:
        """Execute the ReAct (Reason-Act-Observe) loop until the LLM stops
        calling tools or max_iterations is reached."""
        system_prompt, user_prompt = apply_prompt_overlay(
            self.config.system_prompt, self.config.user_prompt
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        self.conversation_flow = [
            {"type": "system_message", "content": system_prompt},
            {"type": "user_message", "content": user_prompt},
        ]
        conversation_flow = self.conversation_flow
        tools_used = self.tools_used
        tool_results = self.tool_results

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")

            # Reason: ask the LLM what to do next
            response = await self.llm_client.invoke_with_tools(
                messages, self.available_tools
            )
            reasoning_details = (getattr(response, "additional_kwargs", None) or {}).get(
                "reasoning_details"
            )
            if reasoning_details:
                logger.debug(f"Preserving reasoning_details for next turn ({len(reasoning_details)} items)")

            messages.append(response)

            usage_metadata = (
                response.usage_metadata if hasattr(response, "usage_metadata") else {}
            )
            response_metadata = (
                response.response_metadata
                if hasattr(response, "response_metadata")
                else {}
            )

            conversation_flow.append(
                {
                    "type": "ai_message",
                    "content": get_text_content(response.content),
                    "usage_metadata": usage_metadata,
                    "response_metadata": response_metadata,
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in (response.tool_calls or [])
                    ],
                }
            )

            logger.info(f"LLM Response: {get_text_content(response.content)}")

            # Terminate if the LLM decided no further tool calls are needed
            if not response.tool_calls or len(response.tool_calls) == 0:
                logger.info("No tool calls requested. Task complete.")
                break

            # Act + Observe: execute each tool call and feed results back
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.debug(f"Tool arguments: {tool_args}")

                exec_result = await self._execute_tool_call(tool_name, tool_args)
                tool_result = exec_result["result"]
                target_gym = exec_result["gym_server"]

                logger.info(f"Tool result success: {tool_result.get('success')}")

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

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result.get("result", {})),
                        tool_call_id=tool_call.get("id", ""),
                    )
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
            "final_response": get_text_content(messages[-1].content) if messages else "",
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }
