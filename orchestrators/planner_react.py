"""Two-pass planner + ReAct orchestrator.

Pass 1: A meta/planner LLM generates a strategic execution plan.
Pass 2: An executor LLM runs the standard ReAct loop, but with the original
        user task augmented by the generated plan.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from benchmark.llm_client import LLMClient
from .base import AgentOrchestrator

logger = logging.getLogger(__name__)


# ============================================================================
# PLAN GENERATOR
# ============================================================================


class SimplePlanGenerator:
    """Generates execution plans using a meta agent LLM."""

    def __init__(self, planner_llm, prompt_template: Optional[str] = None):
        self.planner_llm = planner_llm
        self.prompt_template = prompt_template or self._default_prompt_template()

    @staticmethod
    def _default_prompt_template() -> str:
        """Default prompt template for plan generation."""
        return """You are an enterprise planning agent responsible for creating high-level execution plans. Your role is to analyze the task context and provide strategic guidance for a downstream executor agent.

# Input Context

## System Policy (HIGHEST PRIORITY)
{system_policy}

## User Task
{user_task}

## Available Tools
{tools}

# Planning Instructions

Analyze the above context and create a comprehensive execution plan that addresses the following dimensions:

## 1. User Intent & Tool Alignment
- What is the user trying to accomplish? Identify the core intent beyond the literal request.
- Which tools are most relevant for achieving this intent?
- How should the tools be sequenced or combined to accomplish the goal?

## 2. System Policy Compliance
- Which system policies are relevant to this task?
- Are there any conflicts between the user task and system policies?
- If conflicts exist, how should the executor prioritize system policies over user requests?
- What constraints or guardrails must be maintained throughout execution?

## 3. Risk Assessment & Side Effects
- Are there potential side effects of executing this plan on the broader system?
- Does the system policy require any specific handling or additional considerations (such as database integrity etc)?

# Output Format

Provide your plan in the following structure:

**INTENT ANALYSIS:**
[Describe the core user intent and how it maps to the available tools]

**RELEVANT POLICIES:**
[List applicable system policies and any policy conflicts with the user request]

**EXECUTION STRATEGY:**
[Provide a high-level plan of attack with key steps, tool usage, and decision points]

**RISK MITIGATION:**
[Identify potential side effects and safety measures the executor should observe]

**CRITICAL NOTES:**
[Any additional constraints, priorities, or important considerations]

Remember: This is a strategic plan, not a detailed execution trace. The executor agent will explore the environment and make tactical decisions while following this plan."""

    def construct_prompt(
        self, system_prompt: str, user_prompt: str, tools: List[Dict[str, Any]]
    ) -> str:
        return self.prompt_template.format(
            system_policy=system_prompt,
            user_task=user_prompt,
            tools=json.dumps(tools, indent=2),
        )

    async def generate_plan(
        self, system_prompt: str, user_prompt: str, tools: List[Dict[str, Any]]
    ) -> str:
        prompt = self.construct_prompt(system_prompt, user_prompt, tools)

        logger.info("🧠 Meta Agent: Generating execution plan...")
        response = await self.planner_llm.ainvoke([HumanMessage(content=prompt)])
        plan = response.content

        logger.info(f"✅ Plan generated ({len(plan)} characters)")
        logger.debug(f"Generated plan:\n{plan}")

        return plan


# ============================================================================
# PLANNER + REACT ORCHESTRATOR
# ============================================================================


class PlannerReactOrchestrator(AgentOrchestrator):
    """
    Two-pass orchestrator:
    1. A planner LLM generates a strategic execution plan.
    2. The executor LLM runs a ReAct loop guided by that plan.

    Args:
        planner_llm_client: LLMClient used for plan generation. Defaults to
                            the executor's llm_client when not provided.
        prompt_template:    Optional custom template for the planner prompt.
    """

    def __init__(
        self,
        *args,
        planner_llm_client: Optional["LLMClient"] = None,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._planner_llm_client = planner_llm_client or self.llm_client
        self._prompt_template = prompt_template
        self._last_plan: Optional[str] = None  # stored for get_result_metadata()

    def get_result_metadata(self) -> Dict[str, Any]:
        """Surface the generated plan in the run result."""
        if self._last_plan is not None:
            return {"generated_plan": self._last_plan}
        return {}

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the two-pass planner + ReAct strategy.

        Pass 1 — Plan generation:
            The planner LLM analyses the system prompt, user prompt, and
            available tools to produce a strategic execution plan.

        Pass 2 — Plan-guided ReAct loop:
            The executor LLM receives the original task augmented with the
            generated plan and runs the standard Reason-Act-Observe loop.
        """
        # ------------------------------------------------------------------
        # PASS 1: Generate execution plan
        # ------------------------------------------------------------------
        logger.info(f"\n{'='*80}")
        logger.info("PASS 1: META AGENT - PLAN GENERATION")
        logger.info(f"{'='*80}\n")

        planner_llm = self._planner_llm_client.llm.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=3,
        )
        plan_generator = SimplePlanGenerator(
            planner_llm=planner_llm,
            prompt_template=self._prompt_template,
        )
        plan = await plan_generator.generate_plan(
            system_prompt=self.config.system_prompt,
            user_prompt=self.config.user_prompt,
            tools=self.available_tools,
        )
        self._last_plan = plan

        # ------------------------------------------------------------------
        # PASS 2: Plan-guided ReAct loop
        # ------------------------------------------------------------------
        logger.info(f"\n{'='*80}")
        logger.info("PASS 2: EXECUTOR AGENT - REACT LOOP WITH PLAN")
        logger.info(f"{'='*80}\n")

        enhanced_user_message = f"""# Original Task
{self.config.user_prompt}

# Strategic Plan (from Meta Agent)
{plan}

Please execute the task following the strategic plan above. Use the available tools to accomplish the user's request while adhering to the system policy and the guidance provided in the plan."""

        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=enhanced_user_message),
        ]

        conversation_flow = [
            {"type": "system_message", "content": self.config.system_prompt},
            {
                "type": "user_message",
                "content": enhanced_user_message,
                "original_task": self.config.user_prompt,
                "generated_plan": plan,
            },
        ]
        tools_used = []
        tool_results = []

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            response = await self.llm_client.invoke_with_tools(
                messages, self.available_tools
            )

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
                    "content": response.content,
                    "usage_metadata": usage_metadata,
                    "response_metadata": response_metadata,
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in (response.tool_calls or [])
                    ],
                }
            )

            logger.info(f"LLM Response: {response.content}")

            messages.append(response)

            if not response.tool_calls or len(response.tool_calls) == 0:
                logger.info("No tool calls requested. Task complete.")
                break

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
            "final_response": messages[-1].content if messages else "",
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }
