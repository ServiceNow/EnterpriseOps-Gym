"""Decomposing planner orchestrator.

Three-phase multi-agent architecture:
  Phase 1 — Plan generation: A meta/planner LLM decomposes the task into 2–5
             sequential subtasks, each with a clear objective and dependencies.
  Phase 2 — Sequential execution: An executor sub-agent runs an independent
             ReAct loop for each subtask. A shared WorkingMemory accumulates
             key findings and passes them forward as context.
  Phase 3 — Aggregation: Results are combined into a unified output.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from benchmark.llm_client import LLMClient
from .base import AgentOrchestrator

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def extract_json_from_llm_response(content: str) -> str:
    """Extract JSON from an LLM response, handling markdown code fences."""
    content = content.strip()

    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()

    if "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()

    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            return content[start:end + 1]

    if content.startswith("{") and content.endswith("}"):
        return content

    raise ValueError("Could not extract JSON from response")


def extract_usage_from_response(response: Any) -> Dict[str, int]:
    """
    Extract token usage from a LangChain response object.

    Handles provider-specific formats:
    - Anthropic / AWS Bedrock: response.usage_metadata or response_metadata['usage']
    - OpenAI: response_metadata['token_usage']
    """
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            if isinstance(response.usage_metadata, dict):
                usage["input_tokens"] = response.usage_metadata.get("input_tokens", 0)
                usage["output_tokens"] = response.usage_metadata.get("output_tokens", 0)
            else:
                usage["input_tokens"] = getattr(response.usage_metadata, "input_tokens", 0)
                usage["output_tokens"] = getattr(response.usage_metadata, "output_tokens", 0)
            return usage

        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage" in metadata:
                usage_data = metadata["usage"]
                usage["input_tokens"] = usage_data.get("input_tokens", 0)
                usage["output_tokens"] = usage_data.get("output_tokens", 0)
                return usage
            if "token_usage" in metadata:
                token_usage = metadata["token_usage"]
                usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
                usage["output_tokens"] = token_usage.get("completion_tokens", 0)
                return usage

    except Exception as e:
        logger.warning(f"Failed to extract usage from response: {e}")

    return usage


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class SubTask:
    """Represents a single subtask in the decomposed plan."""

    id: int
    title: str
    description: str
    rationale: str
    dependencies: List[int] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    expected_outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubTaskResult:
    """Result from executing a single subtask."""

    subtask_id: int
    title: str
    success: bool
    summary: str
    structured_output: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    conversation_length: int = 0
    raw_conversation: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationResult:
    """Complete result from an orchestrated multi-agent execution."""

    overall_success: bool
    plan: str
    subtasks: List[SubTask]
    subtask_results: List[SubTaskResult]
    final_output: str
    total_tools_used: List[str]
    total_iterations: int
    working_memory: Dict[str, Any] = field(default_factory=dict)
    planner_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_success": self.overall_success,
            "plan": self.plan,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "subtask_results": [sr.to_dict() for sr in self.subtask_results],
            "final_output": self.final_output,
            "total_tools_used": self.total_tools_used,
            "total_iterations": self.total_iterations,
            "working_memory": self.working_memory,
            "planner_usage": self.planner_usage,
        }


# ============================================================================
# PLAN GENERATOR
# ============================================================================


class DecomposingPlanGenerator:
    """Generates an execution plan and decomposes it into 2–5 subtasks."""

    def __init__(self, planner_llm, decomposition_prompt_template: Optional[str] = None):
        self.planner_llm = planner_llm
        self.decomposition_prompt_template = (
            decomposition_prompt_template or self._default_decomposition_template()
        )

    @staticmethod
    def _default_decomposition_template() -> str:
        return """You are a meta agent responsible for creating an execution plan and decomposing it into subtasks.

# Input Context

## System Policy (HIGHEST PRIORITY)
{system_policy}

## User Task
{user_task}

## Available Tools
{tools}

# Your Responsibilities

1. **Analyze the Task**: Understand what the user wants to accomplish
2. **Create Strategic Plan**: Develop a high-level approach
3. **Decompose into Subtasks**: Break the task into 2-5 logical subtasks

# Decomposition Guidelines

- Each subtask should be at a clean logical boundary
- Subtasks should be relatively independent but can depend on previous results
- Number of subtasks: Between 2 and 5 (choose based on complexity)
- Each subtask should have:
  * Clear objective
  * Specific tools it will likely use
  * Expected outcome
  * Dependencies on previous subtasks (if any)

# CRITICAL DEPENDENCY RULE

**Subtasks will be executed in sequential order (1, 2, 3, ...).** Therefore:

- A subtask can ONLY depend on subtasks with LOWER IDs
- Subtask 1 must have no dependencies (dependencies: [])
- Subtask 2 can only depend on [1] or []
- Subtask 3 can only depend on [1], [2], [1, 2], or []
- Subtask N can only depend on subtasks 1 through N-1

**NEVER create forward dependencies** (e.g., Subtask 2 depending on Subtask 3 is INVALID)

Order your subtasks logically so that information flows forward, not backward.

# Output Format

Provide your response in the following JSON structure:

```json
{{
  "strategic_plan": "High-level description of the overall approach",
  "rationale": "Why this decomposition makes sense",
  "subtasks": [
    {{
      "id": 1,
      "title": "Short title for the subtask",
      "description": "Detailed description of what this subtask should accomplish",
      "rationale": "Why this subtask is needed and its role in the overall plan",
      "dependencies": [],
      "required_tools": ["tool1", "tool2"],
      "expected_outcome": "What should be accomplished after this subtask"
    }},
    {{
      "id": 2,
      "title": "Next subtask",
      "description": "...",
      "rationale": "...",
      "dependencies": [1],
      "required_tools": ["tool3"],
      "expected_outcome": "..."
    }},
    {{
      "id": 3,
      "title": "Third subtask",
      "description": "...",
      "rationale": "...",
      "dependencies": [1, 2],
      "required_tools": ["tool4"],
      "expected_outcome": "..."
    }}
  ]
}}
```

**Example of VALID dependencies:**
- Subtask 1: dependencies: []
- Subtask 2: dependencies: [1]
- Subtask 3: dependencies: [1, 2]
- Subtask 4: dependencies: [3]
- Subtask 5: dependencies: [4]

**Example of INVALID dependencies (DO NOT DO THIS):**
- Subtask 2: dependencies: [3] ← WRONG! Can't depend on later subtask
- Subtask 1: dependencies: [2] ← WRONG! First subtask must have no dependencies

# Important Notes

- Dependencies: List the IDs of previous subtasks that this subtask needs information from
- Required Tools: List tools by their exact names as provided in the available tools
- Keep descriptions clear and actionable
- Each subtask should be executable by an independent agent with proper context
- The first subtask typically has no dependencies, later ones may depend on earlier results

Now, analyze the task and provide your structured plan."""

    def construct_prompt(
        self, system_prompt: str, user_prompt: str, tools: List[Dict[str, Any]]
    ) -> str:
        tool_descriptions = [
            f"- {t.get('name')}: {t.get('description', 'No description')}"
            for t in tools
        ]
        return self.decomposition_prompt_template.format(
            system_policy=system_prompt,
            user_task=user_prompt,
            tools="\n".join(tool_descriptions),
        )

    async def generate_plan_and_subtasks(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        max_retries: int = 3,
    ) -> Tuple[str, List[SubTask], Dict[str, int]]:
        """
        Generate a strategic plan and decompose it into subtasks.

        Returns:
            Tuple of (strategic_plan, subtasks, planner_usage)
        """
        prompt = self.construct_prompt(system_prompt, user_prompt, tools)
        logger.info("🧠 Meta Agent: Generating plan and decomposing into subtasks...")

        for attempt in range(max_retries):
            try:
                response = await self.planner_llm.ainvoke([HumanMessage(content=prompt)])

                usage = extract_usage_from_response(response)
                logger.info(
                    f"📊 Planner usage: {usage['input_tokens']} input + "
                    f"{usage['output_tokens']} output = "
                    f"{usage['input_tokens'] + usage['output_tokens']} tokens"
                )

                content = extract_json_from_llm_response(response.content)
                logger.debug(f"Extracted JSON content (attempt {attempt + 1}):\n{content[:500]}")

                parsed = json.loads(content)

                if not isinstance(parsed, dict):
                    raise ValueError(f"Parsed JSON is not a dictionary, got {type(parsed)}")

                logger.debug(f"Parsed keys: {list(parsed.keys())}")

                if "strategic_plan" not in parsed:
                    raise KeyError(
                        f"Missing 'strategic_plan' key. Available keys: {list(parsed.keys())}"
                    )
                if "subtasks" not in parsed:
                    raise KeyError(
                        f"Missing 'subtasks' key. Available keys: {list(parsed.keys())}"
                    )

                strategic_plan = parsed.get("strategic_plan", "")
                rationale = parsed.get("rationale", "")

                subtasks = []
                for st_data in parsed.get("subtasks", []):
                    subtasks.append(SubTask(
                        id=st_data["id"],
                        title=st_data["title"],
                        description=st_data["description"],
                        rationale=st_data.get("rationale", ""),
                        dependencies=st_data.get("dependencies", []),
                        required_tools=st_data.get("required_tools", []),
                        expected_outcome=st_data.get("expected_outcome", ""),
                    ))

                if len(subtasks) == 0:
                    raise ValueError("No subtasks generated")

                if not (2 <= len(subtasks) <= 5):
                    logger.warning(
                        f"⚠️ Subtask count {len(subtasks)} outside recommended range [2, 5]"
                    )

                # Validate that all dependencies point to earlier subtasks
                subtask_ids = {st.id for st in subtasks}
                for st in subtasks:
                    for dep_id in st.dependencies:
                        if dep_id not in subtask_ids:
                            raise ValueError(
                                f"Subtask {st.id} has invalid dependency {dep_id} "
                                f"(subtask {dep_id} does not exist)"
                            )
                        if dep_id >= st.id:
                            raise ValueError(
                                f"Subtask {st.id} has forward/circular dependency on subtask {dep_id}. "
                                f"Subtasks must only depend on earlier subtasks (lower IDs)."
                            )

                full_plan = f"{strategic_plan}\n\nRationale: {rationale}"

                logger.info(f"✅ Plan generated with {len(subtasks)} subtasks")
                logger.info(f"Strategic Plan: {strategic_plan[:200]}...")
                for st in subtasks:
                    logger.info(
                        f"  Subtask {st.id}: {st.title} (depends on: {st.dependencies})"
                    )

                return full_plan, subtasks, usage

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"❌ Attempt {attempt + 1}/{max_retries} failed to parse plan: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying plan generation...")
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        f"Response content: {response.content if 'response' in locals() else 'N/A'}"
                    )
                    raise ValueError(
                        f"Failed to parse structured plan after {max_retries} attempts: {e}"
                    )


# ============================================================================
# WORKING MEMORY
# ============================================================================


class WorkingMemory:
    """
    Shared memory accumulated across sequential subtasks.

    Each subtask can read accumulated findings and write new ones,
    allowing information to flow forward through the execution chain.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        subtask_id: int,
        subtask_title: str,
        updates: Dict[str, Any],
        summary: str,
    ):
        """Merge a subtask's structured output into shared memory."""
        self.data.update(updates)
        self.history.append({
            "subtask_id": subtask_id,
            "subtask_title": subtask_title,
            "summary": summary,
            "contributions": list(updates.keys()),
        })
        logger.info(
            f"  📝 Memory updated with {len(updates)} new entries: {list(updates.keys())}"
        )

    def get_all(self) -> Dict[str, Any]:
        return self.data.copy()

    def to_prompt_string(self) -> str:
        if not self.data:
            return "Working memory is empty. No information has been discovered yet."
        parts = [
            "# Working Memory (Shared Context)\n",
            "The following information has been discovered by previous subtasks:\n",
            "```json",
            json.dumps(self.data, indent=2),
            "```",
        ]
        return "\n".join(parts)

    def get_history_summary(self) -> str:
        if not self.history:
            return "No history yet."
        return "\n".join(
            f"Subtask {e['subtask_id']} ({e['subtask_title']}): {e['summary'][:80]}..."
            for e in self.history
        )


# ============================================================================
# EXECUTOR SUB-AGENT
# ============================================================================


class ExecutorSubAgent:
    """
    Executes a single subtask with an independent ReAct loop.

    Not a subclass of AgentOrchestrator — it is an internal component
    of DecomposingPlannerOrchestrator, instantiated once per subtask.
    """

    def __init__(
        self,
        subtask: SubTask,
        llm_client,
        mcp_clients: Dict[str, Any],
        tool_to_server_mapping: Dict[str, str],
        available_tools: List[Dict[str, Any]],
        system_prompt: str,
        working_memory: WorkingMemory,
    ):
        self.subtask = subtask
        self.llm_client = llm_client
        self.mcp_clients = mcp_clients
        self.tool_to_server_mapping = tool_to_server_mapping
        self.available_tools = available_tools
        self.system_prompt = system_prompt
        self.working_memory = working_memory

    def _generate_subtask_prompt(self) -> str:
        memory_context = self.working_memory.to_prompt_string()
        return f"""# Your Role
You are an executor agent responsible for completing a specific subtask within a larger multi-step plan.

# Your Subtask

**Title**: {self.subtask.title}

**Description**: {self.subtask.description}

**Why This Matters**: {self.subtask.rationale}

**Expected Outcome**: {self.subtask.expected_outcome}

# Shared Context

{memory_context}

Use this shared information as needed. Previous subtasks have already discovered this information.

# Your Task

Execute this subtask using the available tools. Follow these guidelines:

1. **Use the working memory**: Previous subtasks may have discovered relevant information
2. **Focus on this subtask only**: Don't try to solve the entire problem
3. **Use appropriate tools**: You have access to tools - use them as needed
4. **Be thorough**: Complete the subtask fully before finishing
5. **Adhere to system policies**: Always follow the system policies provided

# When You're Done

Once you've completed the subtask, provide a clear summary of what you accomplished and any key findings.

Ready? Begin executing your subtask now."""

    async def execute(self, max_iterations: int = 25) -> SubTaskResult:
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 EXECUTING SUBTASK {self.subtask.id}: {self.subtask.title}")
        logger.info(f"{'='*80}\n")
        logger.info(f"Description: {self.subtask.description}")
        logger.info(f"Expected outcome: {self.subtask.expected_outcome}")

        memory_data = self.working_memory.get_all()
        if memory_data:
            logger.info(
                f"Working memory contains {len(memory_data)} entries: {list(memory_data.keys())}"
            )
        else:
            logger.info("Working memory is empty (first subtask)")
        logger.info("")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._generate_subtask_prompt()),
        ]

        conversation_flow = []
        tools_used = []
        tool_results = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for iteration in range(max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{max_iterations}")

            try:
                response = await self.llm_client.invoke_with_tools(
                    messages, self.available_tools
                )

                iteration_usage = extract_usage_from_response(response)
                total_usage["input_tokens"] += iteration_usage["input_tokens"]
                total_usage["output_tokens"] += iteration_usage["output_tokens"]

                conversation_flow.append({
                    "type": "ai_message",
                    "content": response.content,
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in (response.tool_calls or [])
                    ],
                })

                logger.info(f"  LLM: {response.content[:200]}...")

                messages.append(response)

                if not response.tool_calls or len(response.tool_calls) == 0:
                    logger.info(f"  ✅ Subtask {self.subtask.id} completed (no more tool calls)")
                    break

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    # Route tool call to the correct MCP server
                    target_gym = self.tool_to_server_mapping.get(
                        tool_name, list(self.mcp_clients.keys())[0]
                    )
                    target_client = self.mcp_clients[target_gym]

                    logger.info(f"  🔧 Executing tool: {tool_name} on {target_gym}")

                    tool_result = await target_client.call_tool(tool_name, tool_args)

                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    tool_results.append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "gym_server": target_gym,
                    })

                    messages.append(
                        ToolMessage(
                            content=json.dumps(tool_result.get("result", {})),
                            tool_call_id=tool_call.get("id", ""),
                        )
                    )

                    conversation_flow.append({
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    })

            except Exception as e:
                logger.error(
                    f"❌ Error in subtask {self.subtask.id} iteration {iteration + 1}: {e}"
                )
                return SubTaskResult(
                    subtask_id=self.subtask.id,
                    title=self.subtask.title,
                    success=False,
                    summary=f"Error during execution: {str(e)}",
                    tools_used=tools_used,
                    conversation_length=len(conversation_flow),
                    raw_conversation=conversation_flow,
                    error=str(e),
                    usage=total_usage,
                )

        final_response = messages[-1].content if messages else ""
        summary = final_response or (
            f"Executed {len([c for c in conversation_flow if c['type'] == 'tool_result'])} tool calls"
        )

        memory_updates, extraction_usage = await self._extract_memory_updates(
            messages, tool_results
        )
        total_usage["input_tokens"] += extraction_usage["input_tokens"]
        total_usage["output_tokens"] += extraction_usage["output_tokens"]

        logger.info(f"✅ Subtask {self.subtask.id} completed: {summary[:100]}...")
        logger.info(
            f"📊 Subtask {self.subtask.id} usage: "
            f"{total_usage['input_tokens']} input + {total_usage['output_tokens']} output = "
            f"{total_usage['input_tokens'] + total_usage['output_tokens']} tokens"
        )

        return SubTaskResult(
            subtask_id=self.subtask.id,
            title=self.subtask.title,
            success=True,
            summary=summary,
            structured_output=memory_updates,
            tools_used=tools_used,
            conversation_length=len(conversation_flow),
            raw_conversation=conversation_flow,
            usage=total_usage,
        )

    async def _extract_memory_updates(
        self,
        messages: List,
        tool_results: List[Dict],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Ask the LLM to distil key findings from this subtask into structured memory."""
        actions_summary = []
        for tr in tool_results:
            tool_name = tr.get("tool_name", "unknown")
            result = tr.get("result", {})
            success = result.get("success", False)
            result_data = result.get("result", "")
            actions_summary.append(
                f"- Called {tool_name}: {'✓ Success' if success else '✗ Failed'}\n"
                f"  Result: {json.dumps(result_data, indent=2) if result_data else 'N/A'}"
            )

        actions_text = "\n".join(actions_summary) if actions_summary else "No tool calls made"

        extraction_prompt = f"""# Memory Extraction Task

You just completed the following subtask:

**Subtask**: {self.subtask.title}
**Description**: {self.subtask.description}
**Goal**: {self.subtask.expected_outcome}

## Actions You Took

{actions_text}

## Your Task

Based on your execution above, identify key information that should be saved to **working memory** for future subtasks to use.

### What to Extract

Extract information that answers questions like:
- What IDs or identifiers were discovered? (user_id, order_id, record_id, etc.)
- What records or entities were retrieved?
- What values were computed or calculated?
- What was the status or outcome of operations?
- What data was created, updated, or deleted?

### What NOT to Extract

- Temporary variables or intermediate calculations
- Tool names or execution metadata
- Redundant information already in working memory
- Overly detailed data that future subtasks won't need

### Output Format

Return a JSON object with descriptive keys and the discovered values:

```json
{{
  "user_id": 12345,
  "user_email": "john@example.com",
  "user_role": "admin",
  "permission_validated": true,
  "records_count": 42
}}
```

If there's nothing worth adding to memory, return: `{{}}`

**Provide only the JSON, no explanation.**"""

        try:
            extraction_messages = messages + [HumanMessage(content=extraction_prompt)]
            response = await self.llm_client.llm.ainvoke(extraction_messages)

            extraction_usage = extract_usage_from_response(response)
            content = extract_json_from_llm_response(response.content)
            memory_updates = json.loads(content)

            if not isinstance(memory_updates, dict):
                logger.warning(f"  ⚠️ Memory updates not a dict: {type(memory_updates)}")
                return {}, extraction_usage

            logger.debug(
                f"  Extracted {len(memory_updates)} memory updates: {list(memory_updates.keys())}"
            )
            return memory_updates, extraction_usage

        except json.JSONDecodeError as e:
            logger.warning(f"  ⚠️ Failed to parse memory updates JSON: {e}")
            return {}, {"input_tokens": 0, "output_tokens": 0}
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to extract memory updates: {e}")
            return {}, {"input_tokens": 0, "output_tokens": 0}


# ============================================================================
# DECOMPOSING PLANNER ORCHESTRATOR
# ============================================================================


class DecomposingPlannerOrchestrator(AgentOrchestrator):
    """
    Multi-agent orchestrator that decomposes a task into subtasks and executes
    them sequentially with shared working memory.

    Args:
        planner_llm_client:            LLMClient for plan generation. Defaults to
                                       the executor's llm_client when not provided.
        abort_on_subtask_failure:      Stop execution if any subtask fails.
                                       Defaults to False (continue on failure).
        max_iterations:                Maximum ReAct iterations *per subtask*.
                                       Defaults to 25 (lower than the base default
                                       of 50 because each subtask is scoped).
        decomposition_prompt_template: Optional custom prompt for the planner.
    """

    def __init__(
        self,
        *args,
        planner_llm_client: Optional["LLMClient"] = None,
        abort_on_subtask_failure: bool = False,
        max_iterations: int = 25,
        decomposition_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, max_iterations=max_iterations, **kwargs)
        self._planner_llm_client = planner_llm_client or self.llm_client
        self._abort_on_subtask_failure = abort_on_subtask_failure
        self._decomposition_prompt_template = decomposition_prompt_template
        self._orchestration_result: Optional[OrchestrationResult] = None

    def get_result_metadata(self) -> Dict[str, Any]:
        """Surface orchestration metadata and token usage in the run result."""
        if self._orchestration_result is None:
            return {}
        result = self._orchestration_result
        return {
            "orchestration_metadata": {
                "strategic_plan": result.plan,
                "num_subtasks": len(result.subtasks),
                "subtask_details": [st.to_dict() for st in result.subtasks],
                "subtask_results": [sr.to_dict() for sr in result.subtask_results],
                "overall_success": result.overall_success,
                "total_iterations": result.total_iterations,
                "working_memory": result.working_memory,
            },
            "usage": self._build_usage_structure(result),
        }

    async def execute(self) -> Dict[str, Any]:
        """
        Run the three-phase orchestration:
          Phase 1: Decompose the task into subtasks.
          Phase 2: Execute each subtask sequentially with shared working memory.
          Phase 3: Aggregate all results into a unified output.
        """
        logger.info(f"\n{'='*80}")
        logger.info("🎯 ORCHESTRATED MULTI-AGENT EXECUTION")
        logger.info(f"{'='*80}\n")

        plan_generator = DecomposingPlanGenerator(
            planner_llm=self._planner_llm_client.llm,
            decomposition_prompt_template=self._decomposition_prompt_template,
        )

        # ------------------------------------------------------------------
        # PHASE 1: Plan generation and task decomposition
        # ------------------------------------------------------------------
        logger.info("📋 PHASE 1: Plan Generation and Task Decomposition")
        logger.info("-" * 80)

        try:
            strategic_plan, subtasks, planner_usage = (
                await plan_generator.generate_plan_and_subtasks(
                    system_prompt=self.config.system_prompt,
                    user_prompt=self.config.user_prompt,
                    tools=self.available_tools,
                )
            )
        except Exception as e:
            logger.error(f"❌ Failed to generate plan: {e}")
            failed_result = OrchestrationResult(
                overall_success=False,
                plan=f"Plan generation failed: {e}",
                subtasks=[],
                subtask_results=[],
                final_output=f"Failed to generate plan: {e}",
                total_tools_used=[],
                total_iterations=0,
                planner_usage={"input_tokens": 0, "output_tokens": 0},
            )
            self._orchestration_result = failed_result
            return self._build_execute_return(failed_result)

        # ------------------------------------------------------------------
        # PHASE 2: Sequential subtask execution
        # ------------------------------------------------------------------
        logger.info(f"\n📋 PHASE 2: Sequential Subtask Execution ({len(subtasks)} subtasks)")
        logger.info("-" * 80)

        working_memory = WorkingMemory()
        subtask_results: List[SubTaskResult] = []
        total_tools_used: List[str] = []
        total_iterations = 0

        for subtask in subtasks:
            subagent = ExecutorSubAgent(
                subtask=subtask,
                llm_client=self.llm_client,
                mcp_clients=self.mcp_clients,
                tool_to_server_mapping=self.tool_to_server_mapping,
                available_tools=self.available_tools,
                system_prompt=self.config.system_prompt,
                working_memory=working_memory,
            )

            result = await subagent.execute(max_iterations=self.max_iterations)
            subtask_results.append(result)

            if result.success and result.structured_output:
                working_memory.update(
                    subtask_id=result.subtask_id,
                    subtask_title=result.title,
                    updates=result.structured_output,
                    summary=result.summary,
                )

            for tool in result.tools_used:
                if tool not in total_tools_used:
                    total_tools_used.append(tool)
            total_iterations += result.conversation_length

            if not result.success:
                logger.warning(f"⚠️ Subtask {subtask.id} failed: {result.error}")
                if self._abort_on_subtask_failure:
                    logger.error(
                        "❌ Aborting execution due to subtask failure "
                        "(abort_on_subtask_failure=True)"
                    )
                    break
                else:
                    logger.info("Continuing to next subtask despite failure...")

        # ------------------------------------------------------------------
        # PHASE 3: Result aggregation
        # ------------------------------------------------------------------
        logger.info(f"\n📋 PHASE 3: Result Aggregation")
        logger.info("-" * 80)

        overall_success = all(r.success for r in subtask_results)
        final_output = self._generate_final_output(subtask_results)

        logger.info(f"Overall Success: {overall_success}")
        logger.info(f"Total Subtasks: {len(subtasks)}")
        logger.info(f"Successful Subtasks: {sum(1 for r in subtask_results if r.success)}")
        logger.info(f"Total Tools Used: {len(total_tools_used)}")
        logger.info(f"Total Iterations: {total_iterations}")
        logger.info(f"Working Memory Entries: {len(working_memory.get_all())}")

        orchestration_result = OrchestrationResult(
            overall_success=overall_success,
            plan=strategic_plan,
            subtasks=subtasks,
            subtask_results=subtask_results,
            final_output=final_output,
            total_tools_used=total_tools_used,
            total_iterations=total_iterations,
            working_memory=working_memory.get_all(),
            planner_usage=planner_usage,
        )
        self._orchestration_result = orchestration_result

        return self._build_execute_return(orchestration_result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_execute_return(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Build the standard execute() return dict from an OrchestrationResult."""
        return {
            "final_response": result.final_output,
            "conversation_flow": self._build_conversation_flow(result),
            "tools_used": result.total_tools_used,
            "tool_results": self._extract_all_tool_results(result),
            "messages": [],  # Not applicable in orchestrated approach
        }

    def _build_conversation_flow(self, result: OrchestrationResult) -> List[Dict[str, Any]]:
        flow = [
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
            {
                "type": "meta_agent_plan",
                "plan": result.plan,
                "subtasks": [st.to_dict() for st in result.subtasks],
            },
        ]
        for subtask_result in result.subtask_results:
            flow.append({
                "type": "subtask_start",
                "subtask_id": subtask_result.subtask_id,
                "title": subtask_result.title,
            })
            flow.extend(subtask_result.raw_conversation)
            flow.append({
                "type": "subtask_complete",
                "subtask_id": subtask_result.subtask_id,
                "summary": subtask_result.summary,
                "success": subtask_result.success,
            })
        return flow

    def _extract_all_tool_results(self, result: OrchestrationResult) -> List[Dict[str, Any]]:
        all_results = []
        for subtask_result in result.subtask_results:
            for conv_item in subtask_result.raw_conversation:
                if conv_item.get("type") == "tool_result":
                    all_results.append(conv_item)
        return all_results

    def _generate_final_output(self, results: List[SubTaskResult]) -> str:
        parts = ["# Orchestrated Execution Summary\n"]
        for result in results:
            status = "✓" if result.success else "✗"
            parts.append(f"\n## {status} Subtask {result.subtask_id}: {result.title}")
            parts.append(f"\n{result.summary}\n")
            if result.error:
                parts.append(f"**Error**: {result.error}\n")
        if results:
            parts.append(f"\n## Final Result\n\n{results[-1].summary}")
        return "\n".join(parts)

    def _build_usage_structure(self, result: OrchestrationResult) -> Dict[str, Any]:
        planner_usage = result.planner_usage
        subtasks_usage = {}
        total_subtasks_input = 0
        total_subtasks_output = 0

        for subtask_result in result.subtask_results:
            subtask_id = f"subtask_{subtask_result.subtask_id}"
            subtask_usage = subtask_result.usage
            subtasks_usage[subtask_id] = {
                "input_tokens": subtask_usage.get("input_tokens", 0),
                "output_tokens": subtask_usage.get("output_tokens", 0),
            }
            total_subtasks_input += subtask_usage.get("input_tokens", 0)
            total_subtasks_output += subtask_usage.get("output_tokens", 0)

        meta_agent_usage = {"input_tokens": 0, "output_tokens": 0}

        total_input = (
            planner_usage.get("input_tokens", 0)
            + meta_agent_usage["input_tokens"]
            + total_subtasks_input
        )
        total_output = (
            planner_usage.get("output_tokens", 0)
            + meta_agent_usage["output_tokens"]
            + total_subtasks_output
        )

        logger.info(f"\n📊 Token Usage Summary:")
        logger.info(
            f"  Planner: {planner_usage.get('input_tokens', 0)} in + "
            f"{planner_usage.get('output_tokens', 0)} out"
        )
        logger.info(f"  Subtasks: {total_subtasks_input} in + {total_subtasks_output} out")
        logger.info(
            f"  Total: {total_input} in + {total_output} out = {total_input + total_output} tokens"
        )

        return {
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
            },
            "breakdown": {
                "planner": {
                    "input_tokens": planner_usage.get("input_tokens", 0),
                    "output_tokens": planner_usage.get("output_tokens", 0),
                },
                "meta_agent": meta_agent_usage,
                "subtasks": subtasks_usage,
            },
        }
