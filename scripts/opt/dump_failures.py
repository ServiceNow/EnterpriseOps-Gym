"""Render failed TRAIN tasks (agent vs golden trajectory) to markdown for analysis subagents.

python scripts/opt/dump_failures.py <domain> <attempt_name> [--prev <attempt> ...]

Writes <results>/<attempt>/failure_dumps/<task_id>.md plus lost.txt (tasks that passed in >=1 of the --prev
attempts, i.e. regressions) and rest.txt (everything else). Only train-split tasks are dumped.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from domains import config_path, get  # noqa: E402

args = sys.argv[1:]
prev = args[args.index("--prev") + 1:] if "--prev" in args else []
args = args[:args.index("--prev")] if "--prev" in args else args
dom = get(args[0])
name = args[1]
run_dir = f"{dom['results']}/{name}/run_1"
out_dir = f"{dom['results']}/{name}/failure_dumps"
ids = set(json.load(open(dom["split"]))["train"])
os.makedirs(out_dir, exist_ok=True)


def trunc(x, n=400):
    s = x if isinstance(x, str) else json.dumps(x)
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n}]"


def golden(tid):
    g = json.load(open(os.path.join(dom["bundle"], tid + ".json")))
    runs = [r for r in g.get("results", []) if r.get("overall_success")] or g.get("results", [])
    if not runs:
        return "(no golden run)"
    lines = []
    for step in json.loads(runs[0]["llm_response"]):
        if "tool_calls_made" in step or step.get("type") == "claude_response":
            if step.get("content"):
                lines.append(f"THOUGHT: {trunc(step['content'], 500)}")
            for tc in step.get("tool_calls_made", []):
                res = tc.get("result", {})
                txt = res.get("content", [{}])[0].get("text", res) if isinstance(res, dict) and res.get("content") else res
                lines.append(f"CALL {tc['tool_name']}({json.dumps(tc.get('parameters'))})\n  -> {trunc(txt)}")
    return "\n".join(lines)


def passed_in(attempt):
    out = set()
    for f in glob.glob(f"{dom['results']}/{attempt}/run_1/*.json"):
        j = json.load(open(f))
        if j["statistics"].get("pass_at_1") == 1.0:
            out.add(j["benchmark_config"]["task_id"])
    return out


prev_pass = set().union(*[passed_in(a) for a in prev]) if prev else set()
lost, rest = [], []
for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    d = json.load(open(f))
    tid = d["benchmark_config"]["task_id"]
    if tid not in ids or d["statistics"].get("pass_at_1") == 1.0:
        continue
    r = d["runs"][0]
    out = [f"# {tid}", f"Config (system_prompt / user_prompt / verifiers): {config_path(dom['domain'], tid)}",
           f"Acting identity: {r.get('execution_context', {}).get('gyms', [{}])[0].get('identity')}",
           "", "## User prompt", d["benchmark_config"].get("user_prompt", ""), "", "## Verifiers"]
    for vname, v in (r.get("verification_results") or {}).items():
        out.append(f"- [{'PASS' if v.get('passed') else 'FAIL'}] {vname}: expected={v.get('expected')} actual={v.get('actual')}")
        if not v.get("passed"):
            out.append("```sql\n" + str(v.get("query", v.get("details", ""))) + "\n```")
    out += ["", "## Agent trajectory (target model)"]
    for m in r.get("conversation_flow", []):
        if m["type"] == "ai_message":
            if m.get("content"):
                out.append(f"SAY: {trunc(m['content'], 1500)}")
            for tc in m.get("tool_calls", []):
                out.append(f"CALL {tc['name']}({json.dumps(tc['args'])})")
        elif m["type"] == "tool_result":
            res = m["result"].get("result", m["result"])
            if isinstance(res, dict) and res.get("content"):
                res = res["content"][0].get("text", res)
            out.append(f"  -> {trunc(res)}")
    out += ["", "## Golden trajectory (reference model run shipped with the task)", golden(tid)]
    open(os.path.join(out_dir, tid + ".md"), "w").write("\n".join(out))
    (lost if tid in prev_pass else rest).append(tid + ".md")
open(f"{out_dir}/lost.txt", "w").write("\n".join(lost))
open(f"{out_dir}/rest.txt", "w").write("\n".join(rest))
print(f"wrote {len(lost) + len(rest)} dumps to {out_dir} (lost={len(lost)} rest={len(rest)})")
