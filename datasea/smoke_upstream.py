"""Phase-1 check: run one official task through the *unmodified* upstream
BenchmarkExecutor.execute_benchmark(), substituting only the orchestrator
(via its public orchestrator_class hook) with a scripted one, so no LLM key is needed.

    uv run python -m datasea.smoke_upstream --task_id <id> --script datasea/tasks/scripts/<id>.json

Exercises: DB seeding, MCP connect, tool discovery/filtering, tool calls,
upstream verifiers, and upstream DB cleanup. Also confirms reset by checking a
fresh database is back at seed state afterwards.
"""

import argparse
import asyncio
import json
import logging

from benchmark.executor import BenchmarkExecutor
from benchmark.models import LLMConfig
from orchestrators.base import AgentOrchestrator

from .env import TaskEnvironment
from .tasks import get_task


class ScriptedOrchestrator(AgentOrchestrator):
    script: list = []

    async def execute(self):
        names = [t["name"] for t in self.available_tools]
        print(f"[smoke] tools discovered: {names}")
        tool_results = []
        for step in self.script:
            out = await self._execute_tool_call(step["tool_name"], step["arguments"])
            res = out["result"]
            text = (res.get("result") or {}).get("content", [{}])[0].get("text", "")
            print(f"[smoke] {step['tool_name']} success={res.get('success')} isError={(res.get('result') or {}).get('isError')} -> {text[:160]}")
            tool_results.append({"tool_name": step["tool_name"], "arguments": step["arguments"], "result": res, "gym_server": out["gym_server"]})
        return {"final_response": "Task completed.", "conversation_flow": [], "tools_used": [s["tool_name"] for s in self.script],
                "tool_results": tool_results, "messages": []}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", required=True)
    p.add_argument("--script", required=True)
    args = p.parse_args()

    entry = get_task(args.task_id)
    ScriptedOrchestrator.script = json.load(open(args.script))

    # Build the config exactly as datasea.env does (upstream evaluate.load_config).
    config = TaskEnvironment(entry["config"]).config
    dummy_llm = LLMConfig(llm_provider="openai", llm_model="unused", llm_api_key="unused")
    ex = BenchmarkExecutor(config, llm_config=dummy_llm, orchestrator_class=ScriptedOrchestrator)
    result = await ex.execute_benchmark()
    run = result["runs"][0]
    if run.get("error"):
        raise SystemExit(f"[smoke] run error: {run['error']}")
    for name, v in run["verification_results"].items():
        print(f"[smoke] verifier {'PASS' if v['passed'] else 'FAIL'}: {name} (expected={v.get('expected')!r} actual={v.get('actual')!r})")
    print(f"[smoke] overall_success={run['overall_success']}")

    # Reset check: a fresh environment must fail the verifiers again (seed state restored).
    fresh = TaskEnvironment(entry["config"])
    await fresh.start()
    v = await fresh.verify("", [])
    await fresh.reset()
    print(f"[smoke] fresh env after reset: overall_success={v['overall_success']} (expected False)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
