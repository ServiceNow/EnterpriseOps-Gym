"""Golden-guided orchestrator - trajectory-conditioned ReAct.

Where :class:`GoldenOrchestrator` replays the recorded oracle trajectory
verbatim (no LLM), this orchestrator keeps the golden **tool sequence** fixed
but puts an LLM in the loop to adapt only the *runtime-reference* argument
values (server-minted ids, handles, tokens, ...) so they match what the
**current** run's prior live tool results actually returned.

Motivation: recorded trajectories hardcode ids that were minted when the
trajectory was recorded. On replay the seeded DB mints *different* ids, so every
downstream step that references a recorded id fails. This orchestrator lets the
model substitute the correct new id, observed live from the prior steps'
results, while a substitution guard prevents it from touching any judgment /
semantic value (enums, roles, summaries, colorId, ttl, reminder minutes, ...).

Contract vs :class:`GoldenOrchestrator`:
- Same golden-file resolution (shared :class:`GoldenFileMixin`).
- Same tool sequence and order (step ``i`` must call ``golden[i].tool_name``).
- Requires a working ``--llm_config`` (one LLM call per step); unlike ``golden``
  which is inert and needs no model.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from .base import AgentOrchestrator
from .golden_common import GoldenFileMixin

logger = logging.getLogger(__name__)

# Static final response for guided golden runs (mirrors the golden convention:
# no free-form final-answer LLM turn).
GUIDED_FINAL_RESPONSE = "Guided Golden Run"

# Substrings that indicate a tool returned a content-level error even when the
# transport reported success (MCP ``success: true`` / ``isError: false``).
_ERROR_MARKERS = (
    "does not exist",
    "not found",
    "no such",
    "error",
    "failed",
    "invalid",
)


class GoldenGuidedOrchestrator(GoldenFileMixin, AgentOrchestrator):
    """Runs the golden sequence with per-step LLM argument adaptation."""

    def __init__(self, *args, config_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path
        # Per-step substitution telemetry: each accept/revert the guard makes.
        self._substitutions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def execute(self) -> Dict[str, Any]:
        manual_tool_executions = self._load_manual_tool_executions()

        messages: List[Any] = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self.config.user_prompt),
        ]
        conversation_flow: List[Dict[str, Any]] = [
            {"type": "system_message", "content": self.config.system_prompt},
            {"type": "user_message", "content": self.config.user_prompt},
        ]
        tools_used: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        # Accumulated parsed ``result`` objects from prior LIVE steps. The guard
        # searches these to decide whether a changed value is a real reference.
        live_results: List[Any] = []

        for idx, step in enumerate(manual_tool_executions):
            tool_name = step.get("tool_name")
            golden_args = step.get("arguments") or {}

            if not tool_name:
                logger.warning(
                    f"[GUIDED] Step {idx + 1} has no tool_name; skipping: {step}"
                )
                continue

            logger.info(
                f"\n--- Guided Step {idx + 1}/{len(manual_tool_executions)}: "
                f"{tool_name} ---"
            )

            # 1-3. Ask the LLM for adapted args, forcing THIS tool. Prior live
            # results are passed as plain text (not a tool_use/tool_result
            # chain) so the call is provider-agnostic and needs no matching
            # assistant tool_use block.
            adapted_args = await self._adapt_args_for_step(
                idx, tool_name, golden_args, live_results
            )

            # 4. Guard: revert any change not backed by prior live results.
            guarded_args = self._guard_args(
                idx, tool_name, golden_args, adapted_args, live_results
            )

            # 5. Execute live against the seeded DB.
            exec_result = await self._execute_tool_call(tool_name, guarded_args)
            tool_result = exec_result["result"]
            target_gym = exec_result["gym_server"]

            content_error = self._content_error(tool_result)
            logger.info(
                f"[GUIDED] Tool result success={tool_result.get('success')} "
                f"content_error={content_error}"
            )
            if content_error:
                logger.warning(
                    f"[GUIDED] Step {idx + 1} ({tool_name}) returned a "
                    f"content-level error despite transport success: "
                    f"{content_error}"
                )

            if tool_name not in tools_used:
                tools_used.append(tool_name)

            tool_results.append(
                {
                    "tool_name": tool_name,
                    "arguments": guarded_args,
                    "result": tool_result,
                    "gym_server": target_gym,
                }
            )

            # 6. Feed the live result back so the next step's LLM sees new ids.
            inner_result = tool_result.get("result", {})
            live_results.append(inner_result)
            messages.append(
                ToolMessage(
                    content=json.dumps(inner_result, default=str),
                    tool_call_id=f"golden-step-{idx}",
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
            "final_response": GUIDED_FINAL_RESPONSE,
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "messages": messages,
        }

    def get_result_metadata(self) -> Dict[str, Any]:
        """Surface per-step substitution telemetry for auditing."""
        return {"golden_substitutions": self._substitutions}

    # ------------------------------------------------------------------ #
    # Per-step LLM argument adaptation (forced tool)
    # ------------------------------------------------------------------ #
    async def _adapt_args_for_step(
        self,
        idx: int,
        tool_name: str,
        golden_args: Dict[str, Any],
        live_results: List[Any],
    ) -> Dict[str, Any]:
        """Ask the LLM to adapt golden_args for THIS tool, forcing the tool via
        single-tool filtering + validation. Falls back to golden_args on any
        deviation (no tool call, wrong tool, or empty args).

        The prior live results are embedded as plain text in the instruction
        rather than replayed as a tool_use/tool_result message chain: the model
        only needs to *read* the new ids, and a synthetic tool_result block with
        no matching assistant tool_use block is rejected by strict providers
        (e.g. Anthropic/Bedrock). A clean [system, user] pair keeps this
        provider-agnostic."""
        tool_def = self._find_tool_def(tool_name)
        if tool_def is None:
            logger.warning(
                f"[GUIDED] Tool '{tool_name}' not in available_tools; using "
                f"golden args verbatim."
            )
            return golden_args

        step_messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=self._build_step_instruction(
                tool_name, golden_args, live_results
            )),
        ]

        try:
            response = await self.llm_client.invoke_with_tools(
                step_messages, [tool_def]
            )
        except Exception as e:  # pragma: no cover - network/provider failure
            logger.warning(
                f"[GUIDED] LLM call failed for step {idx + 1} ({tool_name}): "
                f"{e}. Using golden args."
            )
            return golden_args

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning(
                f"[GUIDED] Step {idx + 1}: model returned no tool call for "
                f"'{tool_name}'; using golden args."
            )
            return golden_args

        call = tool_calls[0]
        if call.get("name") != tool_name:
            logger.warning(
                f"[GUIDED] Step {idx + 1}: model called '{call.get('name')}' "
                f"but golden requires '{tool_name}'; using golden args."
            )
            return golden_args

        adapted = call.get("args") or {}
        if not isinstance(adapted, dict) or not adapted:
            logger.warning(
                f"[GUIDED] Step {idx + 1}: model returned empty/invalid args "
                f"for '{tool_name}'; using golden args."
            )
            return golden_args

        return adapted

    def _find_tool_def(self, tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in self.available_tools:
            if tool.get("name") == tool_name:
                return tool
        return None

    def _build_step_instruction(
        self, tool_name: str, golden_args: Dict[str, Any], live_results: List[Any]
    ) -> str:
        if live_results:
            prior = json.dumps(live_results, indent=2, default=str)
            prior_block = (
                f"Results of the tool calls already executed in THIS run "
                f"(in order; these are the LIVE results against the current "
                f"database):\n```json\n{prior}\n```\n\n"
            )
        else:
            prior_block = (
                "No tools have been executed yet in this run (this is the "
                "first step).\n\n"
            )
        return (
            f"You are re-executing a recorded (golden) trajectory step by step "
            f"against a freshly seeded database.\n\n"
            f"{prior_block}"
            f"For THIS step you must call the tool `{tool_name}` exactly once.\n\n"
            f"Here is the golden argument template for this call:\n"
            f"```json\n{json.dumps(golden_args, indent=2, default=str)}\n```\n\n"
            f"Rules:\n"
            f"1. Keep EVERY value exactly as in the template EXCEPT "
            f"runtime-reference values: server-generated identifiers, handles, "
            f"tokens, resource ids, etc.\n"
            f"2. A recorded id in the template may be stale. If one of the prior "
            f"tool results shown above created or returned the resource this "
            f"step refers to, replace the stale id with the id observed in that "
            f"LIVE prior result.\n"
            f"3. Preserve all judgment / semantic values verbatim: enums, roles, "
            f"summaries, descriptions, colorId, ttl, reminder minutes, booleans, "
            f"counts, free text, timestamps.\n"
            f"4. Never invent a value. Only use values present in the template or "
            f"observed in prior live tool results.\n"
            f"Call `{tool_name}` now with the adapted arguments."
        )

    # ------------------------------------------------------------------ #
    # Substitution guard
    # ------------------------------------------------------------------ #
    def _guard_args(
        self,
        idx: int,
        tool_name: str,
        golden_args: Any,
        adapted_args: Any,
        live_results: List[Any],
    ) -> Any:
        """Recursively reconcile adapted args against golden args.

        A changed leaf value is ACCEPTED only if its string form appears in the
        accumulated prior live results; otherwise it is REVERTED to the golden
        value. Structure (keys / list length) always follows golden to keep the
        call shape faithful to the trajectory.
        """
        # Precompute searchable forms of the prior live results once per step.
        live_values = set()
        for r in live_results:
            self._collect_scalar_strings(r, live_values)
        live_blob = json.dumps(live_results, default=str)

        return self._guard_node(
            idx, tool_name, "", golden_args, adapted_args, live_values, live_blob
        )

    def _guard_node(
        self,
        idx: int,
        tool_name: str,
        path: str,
        golden: Any,
        adapted: Any,
        live_values: set,
        live_blob: str,
    ) -> Any:
        # Dict: follow golden keys; recurse where adapted supplies the same key.
        if isinstance(golden, dict):
            if not isinstance(adapted, dict):
                return golden
            out = {}
            for key, gval in golden.items():
                child_path = f"{path}.{key}" if path else key
                if key in adapted:
                    out[key] = self._guard_node(
                        idx, tool_name, child_path, gval, adapted[key],
                        live_values, live_blob,
                    )
                else:
                    out[key] = gval
            return out

        # List: follow golden length; recurse element-wise where possible.
        if isinstance(golden, list):
            if not isinstance(adapted, list):
                return golden
            out = []
            for i, gval in enumerate(golden):
                child_path = f"{path}[{i}]"
                if i < len(adapted):
                    out.append(self._guard_node(
                        idx, tool_name, child_path, gval, adapted[i],
                        live_values, live_blob,
                    ))
                else:
                    out.append(gval)
            return out

        # Leaf: accept a change only if it is backed by prior live results.
        if adapted == golden:
            return golden

        if self._value_in_live(adapted, live_values, live_blob):
            self._record_substitution(
                idx, tool_name, path, golden, adapted, "accepted"
            )
            logger.info(
                f"[GUIDED][GUARD] ACCEPT step {idx + 1} {tool_name} "
                f"field='{path}': {golden!r} -> {adapted!r} (found in live results)"
            )
            return adapted

        self._record_substitution(
            idx, tool_name, path, golden, adapted, "reverted"
        )
        logger.info(
            f"[GUIDED][GUARD] REVERT step {idx + 1} {tool_name} "
            f"field='{path}': model proposed {adapted!r}, not in live results; "
            f"reverting to golden {golden!r}"
        )
        return golden

    def _value_in_live(
        self, value: Any, live_values: set, live_blob: str
    ) -> bool:
        """True if ``value`` appears in prior live results (structured match
        preferred, substring fallback for embedded ids)."""
        if value is None:
            return False
        # Structured: exact scalar match anywhere in the parsed prior results.
        if value in live_values:
            return True
        s = value if isinstance(value, str) else json.dumps(value, default=str)
        if not s:
            return False
        # Substring fallback: catches ids embedded inside larger result strings.
        if isinstance(value, str) and s in live_blob:
            return True
        return False

    def _collect_scalar_strings(self, obj: Any, out: set) -> None:
        """Recursively collect scalar leaf values (as-is) from a parsed result.

        MCP tool results are frequently ``{content: [{type: text, text: "..."}]}``
        where ``text`` is itself a JSON string; parse those so nested ids become
        first-class searchable values."""
        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_scalar_strings(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_scalar_strings(v, out)
        elif isinstance(obj, str):
            out.add(obj)
            # Attempt to parse embedded JSON (e.g. content[].text payloads).
            stripped = obj.strip()
            if stripped and stripped[0] in "{[":
                try:
                    parsed = json.loads(stripped)
                except (ValueError, TypeError):
                    return
                self._collect_scalar_strings(parsed, out)
        elif isinstance(obj, (int, float, bool)):
            out.add(obj)

    def _record_substitution(
        self,
        idx: int,
        tool_name: str,
        path: str,
        golden_value: Any,
        new_value: Any,
        source: str,
    ) -> None:
        self._substitutions.append(
            {
                "step": idx + 1,
                "tool": tool_name,
                "field": path,
                "golden_value": golden_value,
                "new_value": new_value,
                "source": source,
            }
        )

    # ------------------------------------------------------------------ #
    # Content-level error detection (result.success can be misleading)
    # ------------------------------------------------------------------ #
    def _content_error(self, tool_result: Dict[str, Any]) -> Optional[str]:
        """Return the offending text if the result content looks like an error,
        else None. Inspects text content, not just the transport success flag."""
        inner = tool_result.get("result")
        texts: List[str] = []
        self._collect_text(inner, texts)
        for text in texts:
            low = text.lower()
            if any(marker in low for marker in _ERROR_MARKERS):
                return text[:300]
        return None

    def _collect_text(self, obj: Any, out: List[str]) -> None:
        if isinstance(obj, dict):
            # MCP content blocks: {"type": "text", "text": "..."}
            if obj.get("type") == "text" and isinstance(obj.get("text"), str):
                out.append(obj["text"])
            for v in obj.values():
                self._collect_text(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_text(v, out)
