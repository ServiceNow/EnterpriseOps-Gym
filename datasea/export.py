"""Export trajectories.

    python -m datasea.export [--out_dir datasea/runs/exports] [--include_pilot]

Writes:
    sft_passed.jsonl          tool-calling SFT records (verifier-passed sessions only)
    raw_<status>.jsonl        full raw session + steps, one file per status (nothing dropped)

Pilot-only sessions are excluded from the SFT file unless --include_pilot is set;
they always appear in the raw files with pilot_only=true.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from . import RUNTIME_DIR
from .store import PASSED, Store

DEFAULT_FINAL = "Task completed."


def _tool_message_content(step: Dict[str, Any]) -> str:
    # Matches orchestrators/react.py: ToolMessage(content=json.dumps(tool_result.get("result", {}))),
    # falling back to the error text when the MCP call itself failed.
    res = step.get("result") or {}
    if res.get("result") is not None:
        return json.dumps(res["result"])
    return json.dumps({"error": step.get("error") or res.get("error") or "unknown error"})


def _duration_seconds(session: Dict[str, Any]) -> float:
    if not session.get("ended_at"):
        return None
    return (datetime.fromisoformat(session["ended_at"]) - datetime.fromisoformat(session["started_at"])).total_seconds()


def to_sft_record(session: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": session["system_prompt"]},
        {"role": "user", "content": session["user_prompt"]},
    ]
    for s in steps:
        call_id = f"call_{s['step']}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": s["tool_name"], "arguments": json.dumps(s["arguments"])}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "name": s["tool_name"], "content": _tool_message_content(s)})
    messages.append({"role": "assistant", "content": (session.get("final_response") or "").strip() or DEFAULT_FINAL})

    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}}
        for t in session["available_tools"]
    ]
    verifier = session.get("verifier") or {}
    return {
        "messages": messages,
        "tools": tools,
        "metadata": {
            "session_id": session["session_id"],
            "task_id": session["task_id"],
            "worker_id": session["worker_id"],
            "domain": session["domain"],
            "pilot_only": bool(session["pilot_only"]),
            "verifier_pass": bool(session.get("verifier_pass")),
            "verification_summary": verifier.get("verification_summary"),
            "duration_seconds": _duration_seconds(session),
            "num_tool_calls": len(steps),
            "num_tool_errors": sum(1 for s in steps if s.get("error")),
            "started_at": session["started_at"],
            "ended_at": session["ended_at"],
            "provenance": session["provenance"],
            "environment": session["environment"],
        },
    }


def export(out_dir: str, include_pilot: bool = False) -> Dict[str, int]:
    store = Store()
    os.makedirs(out_dir, exist_ok=True)
    raw_by_status = defaultdict(list)
    sft = []
    for row in store.list_sessions():
        session = store.get_session(row["session_id"])
        steps = store.get_steps(row["session_id"])
        raw_by_status[session["status"]].append({"session": session, "steps": steps})
        if session["status"] == PASSED and (include_pilot or not session["pilot_only"]):
            sft.append(to_sft_record(session, steps))

    counts = {}
    with open(os.path.join(out_dir, "sft_passed.jsonl"), "w") as f:
        for r in sft:
            f.write(json.dumps(r) + "\n")
    counts["sft_passed"] = len(sft)
    for status, items in raw_by_status.items():
        with open(os.path.join(out_dir, f"raw_{status}.jsonl"), "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        counts[f"raw_{status}"] = len(items)
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(RUNTIME_DIR, "exports"))
    p.add_argument("--include_pilot", action="store_true", help="Include pilot_only (public benchmark) tasks in the SFT file")
    args = p.parse_args()
    for k, v in export(args.out_dir, args.include_pilot).items():
        print(f"{k}: {v}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
