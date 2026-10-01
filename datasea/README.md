# DataSea human-demonstration layer for EnterpriseOps-Gym

Nontechnical workers complete EnterpriseOps-Gym tasks in a browser while every
MCP tool call is proxied, logged, verified with the upstream verifiers, and
exported for SFT.

```
Browser ──> DataSea backend (datasea/server.py) ──> EnterpriseOps MCP server (docker) ──> per-session seeded DB
```

Everything lives in `datasea/`. Upstream code is untouched except for one
optional-dependency group added to `pyproject.toml` (`datasea`).

What is reused from upstream, unmodified:

| Concern | Upstream piece |
|---|---|
| Task config parsing | `evaluate.load_config` (same HF→config conversion as `evaluate.py`) |
| Fresh DB per session | `benchmark.mcp_client.create_database_from_file` |
| MCP protocol | `benchmark.mcp_client.MCPClient` |
| Tool discovery + `selected_tools` filtering | `BenchmarkExecutor._discover_and_merge_tools` |
| Verification | `BenchmarkExecutor._run_verifiers` → `VerifierEngine` |
| Reset | `benchmark.mcp_client.delete_database` |

## 1. Start EnterpriseOps (Email domain)

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a Docker runtime.
On Apple Silicon the images are amd64-only; Colima with Rosetta works:

```bash
brew install colima docker
colima start --vm-type vz --vz-rosetta --cpu 4 --memory 8

unzip gym_dbs.zip                                   # seed SQL snapshots
uv sync --extra openai --extra datasea              # openai extra only needed for smoke test / LLM-judge verifiers

docker pull --platform linux/amd64 shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest
docker run -d --name eog-email --platform linux/amd64 -p 8004:8005 \
    shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest
```

Containers listen on 8005 internally (calendar: 8003); map them to the host
ports in `conf.example/ray/domain_conf.json` (email 8004, teams 8002, csm 8001,
calendar 8003, itsm 8006, hr 8008, drive 8009), because task configs hard-code those URLs.

Check the upstream pipeline end-to-end without an LLM (scripted orchestrator
plugged into the unmodified `BenchmarkExecutor.execute_benchmark()`):

```bash
uv run python -m datasea.smoke_upstream \
    --task_id task_20260106_054515_137_1628b966_fe1068d4 \
    --script datasea/tasks/scripts/task_20260106_054515_137_1628b966_fe1068d4.json
```

## 2. Choose pilot tasks

```bash
uv run python -m datasea.tasks import --domain email --task_id <task_id> [--task_id <task_id> ...]
uv run python -m datasea.tasks list
```

Tasks come from `ServiceNow-AI/EnterpriseOps-Gym` at a pinned revision SHA and
are written to `datasea/tasks/pilot_tasks.jsonl`. Every public task is stamped
`pilot_only: true`; it must never be part of a claimed untouched evaluation set.

## 3. Start the human wrapper

```bash
uv run uvicorn datasea.server:app --port 8700      # binds 127.0.0.1 only
```

- Worker UI: <http://localhost:8700/>: enter a worker ID, pick a task, use the generated forms, then Submit or report the task as unclear/broken.
- Admin UI: <http://localhost:8700/admin>: workers, sessions, pass/fail, duration, tool-call count, raw trajectory, verifier output, export button.

Workers see exactly what an agent under evaluation sees: the task's system prompt (as
"Assistant policy"), the user request, and the task's MCP tools. They never see
verifiers, expected values, seed files, auth tokens, or pass/fail.

Forms are generated from each tool's MCP `inputSchema` (`static/schema_form.js`):
enums → dropdowns, booleans → Yes/No, arrays/objects → repeatable/nested groups,
`$ref`/`anyOf`/nullable unions resolved, regex `pattern` validated, and time-like
fields get a UTC date picker that can emit ISO 8601, epoch ms, or epoch seconds.

## 4. Session lifecycle and reset

1. **Start**: seed a brand-new database from the task's SQL snapshot (unique `database_id`).
2. **Each action**: the backend calls MCP, then appends a step (timestamp, tool, exact args, exact result, error, latency) to SQLite.
3. **Submit / Unclear**: run the upstream verifiers on the final DB state, store the full verifier output, and set the status to `passed`, `failed`, or `flagged_unclear`.
4. **Reset**: delete the session database. The next session always starts from a fresh seed.

Abandoned live sessions can be reset from the admin page (status `error`,
trajectory kept). On server shutdown, all live session DBs are deleted.
Nothing is ever removed from `datasea/runs/datasea.sqlite`.

## 5. Export

```bash
uv run python -m datasea.export                    # excludes pilot_only tasks from SFT file
uv run python -m datasea.export --include_pilot    # include them (pilot engineering use)
```

Writes to `datasea/runs/exports/`:

- `sft_passed.jsonl`: one record per verifier-passed session: OpenAI-style
  `messages` (system, user, then an assistant `tool_calls` message and a `tool` message per action,
  then the final assistant reply, defaulting to "Task completed."), the `tools` list, and `metadata`
  (task, worker, domain, pilot_only, verifier summary, duration, counts, repo
  commit, HF revision, docker image digest, seed SHA-256).
  Tool message content uses the same serialization as `orchestrators/react.py`.
- `raw_<status>.jsonl`: the full raw session and steps for every status (passed, failed,
  flagged_unclear, error), independent of any chat format.

## Files

```
datasea/
  env.py            live per-session environment (upstream seeding/MCP/verifier/reset)
  server.py         FastAPI backend: worker + admin APIs, logging proxy
  store.py          SQLite sessions + steps
  tasks.py          HF import → pilot catalog, worker-safe view
  export.py         SFT + raw JSONL export
  provenance.py     git commit, docker digest, seed hashes
  smoke_upstream.py Phase-1 upstream pipeline check
  static/           worker.html, admin.html, schema_form.js, style.css
  tasks/            pilot_tasks.jsonl, scripts/ (smoke-test action scripts)
  runs/             (gitignored) sqlite DB + exports
```
