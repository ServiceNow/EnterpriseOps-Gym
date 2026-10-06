# Running Evaluations

How to evaluate a vendor task bundle for **any domain** with **any model** in
`conf/llm/`. This covers evaluation only — producing traces and a pass rate. To
classify failures as benchmark defects vs. agent mistakes, continue with
[`SAMPLE_AUDIT.md`](SAMPLE_AUDIT.md).

```
vendor bundle ──► prepare_configs.py ──► evaluate.py ──► traces + pass rate
 task_*.json        flat configs          one per task
 db_seeds/*.sql
```

Budget ~20–45 min per model for 180 tasks at `--concurrency 4`.

---

## The three variables

Everything below is parameterized by these. Pick them once and keep them
consistent — mismatches are the main source of silent failure.

| Variable | Example | Where it comes from |
|---|---|---|
| `DOMAIN` | `hr` | A key in `conf/ray/domain_conf.json` |
| `BUNDLE` | `delivery_batch_09-30` | The vendor bundle directory |
| `MODEL` | `gpt_5p6_luna_high` | A filename stem in `conf/llm/` |

```bash
DOMAIN=hr
BUNDLE=delivery_batch_09-30
MODEL=gpt_5p6_luna_high
```

### Domain → MCP server

From `conf/ray/domain_conf.json`. The **host port** is what matters; every
container listens internally on **8005**.

| Domain | MCP name | Host port |
|---|---|---:|
| `csm` | `sn-csm-server` | 8001 |
| `teams` | `gym-teams-mcp` | 8002 |
| `calendar` | `gym-calendar` | 8003 |
| `email` | `gym-email-mcp` | 8004 |
| `itsm` | `gym-itsm-mcp` | 8006 |
| `hr` | `sn-hr-internal` | 8008 |
| `drive` | `gym-google-drive-mcp` | 8009 |

This file is the single source of truth — re-read it rather than trusting this
table, and note the ports are **not** contiguous or alphabetical.

### Available models

`ls conf/llm/`. As of this writing: `claude_opus_5`, `claude_opus_5_high`,
`gpt_5p6_luna`, `gpt_5p6_luna_high`, `gpt_5p6_sol`, `gpt_5p6_terra`,
`gpt_5p6_terra_high`, `gpt_6p1_sol`. The `_high` suffix means high thinking
effort. (`my-model.json` and `openrouter.json` are templates, not real models.)

Model configs may contain credentials. Don't read or print them; pass the path.

---

## 1. Start the MCP server

Vendor images arrive as a tar (`sn-hr.tar`), not from a registry.

```bash
docker load -i sn-$DOMAIN.tar                  # → Loaded image: sn-hr:latest
docker run -d --name sn-$DOMAIN-server -p 8008:8005 sn-$DOMAIN:latest
curl -s http://localhost:8008/health           # → {"status":"healthy", ...}
```

Replace `8008` with the host port for your domain from the table above. The
**container name is arbitrary**; only the port binding has to be right, because
that is what `prepare_configs.py` bakes into every config as
`mcp_server_url`. Confirm the internal port with:

```bash
docker inspect sn-$DOMAIN:latest --format '{{json .Config.ExposedPorts}}'
```

If a server is already running, just health-check it:

```bash
docker ps --format '{{.Names}}\t{{.Ports}}'
```

On Apple Silicon an `amd64` image runs under emulation — functional, slower.

---

## 2. Convert the bundle to evaluator configs

Vendor JSON is **not** what `evaluate.py` reads: it nests fields under
`config`, uses `gym_server_id` instead of a server name/URL, `verifier_configs`
instead of `verifiers`, and carries no seed path. Convert it:

```bash
.venv/bin/python prepare_configs.py \
  --input_dir  "$BUNDLE" \
  --output_dir ".local_configs/${DOMAIN}_${BUNDLE}" \
  --mode oracle --domain "$DOMAIN" --clean
```

Output: one `oracle__<domain>__<task_id>.json` per task, with the MCP
name/URL resolved from `domain_conf.json`, an absolute `seed_database_file`
path attached, verifiers reshaped, and each verifier's target gym remapped to
the server name.

**Read the last line.** Skipped tasks mean the evaluation silently covers fewer
tasks than were delivered:

```
Wrote 180 configs to .local_configs/hr_delivery_batch_09-30 (0 skipped)
```

`--mode` and `--domain` are **only labels used in output filenames** — `--mode
oracle` does *not* mean oracle/replay mode, these are real LLM runs. The labels
matter anyway, because they propagate into the result filenames
(`results_oracle__hr__<task_id>.json`) and `auto-detect`'s adapter rejects any
trace whose filename lacks the `__<domain>__` segment it was invoked with. Use
the same `--domain` for both steps.

`--seed_dir` defaults to `<input_dir>/db_seeds`; pass it only if the seeds live
elsewhere.

---

## 3. Evaluate

```bash
.venv/bin/python evaluate.py \
  --configs_folder ".local_configs/${DOMAIN}_${BUNDLE}" \
  --llm_config "conf/llm/${MODEL}.json" \
  --output_folder "results/react/${MODEL}/${DOMAIN}/${BUNDLE}" \
  --orchestrator react --concurrency 4 --num_runs 1
```

Traces land in `…/run_1/results_oracle__<domain>__<task_id>.json`.

**Use `.venv/bin/python`, never system `python3`.** The system interpreter may
lack `langchain_openai` / `langchain_anthropic`; every task then fails with
`ModuleNotFoundError` while `evaluate.py` still exits 0.

Several models, one domain — **sequentially**, not in parallel:

```bash
for MODEL in gpt_5p6_luna_high claude_opus_5_high; do
  .venv/bin/python evaluate.py \
    --configs_folder ".local_configs/${DOMAIN}_${BUNDLE}" \
    --llm_config "conf/llm/${MODEL}.json" \
    --output_folder "results/react/${MODEL}/${DOMAIN}/${BUNDLE}" \
    --orchestrator react --concurrency 4 --num_runs 1
done
```

Models share one MCP container; running them concurrently doubles the database
create/seed/delete load and produces spurious failures.

Other orchestrators (`planner_react`, `decomposing`) additionally require
`--planner_llm_config`. `--hf_dataset` replaces `--configs_folder` for pulling
tasks from HuggingFace instead of a local bundle, and is the only situation in
which `evaluate.py`'s own `--domain`/`--mode` flags do anything.

---

## 4. Verify the run — the exit code is not a success signal

`evaluate.py` swallows per-task exceptions and **exits 0 even if every task
failed**. Three things must hold.

```bash
ls ".local_configs/${DOMAIN}_${BUNDLE}"/*.json | wc -l     # task count
ls "results/react/${MODEL}/${DOMAIN}/${BUNDLE}/run_1"/*.json | wc -l   # must match
```

```bash
.venv/bin/python -I - <<'EOF'
import json, glob, os
d = f"results/react/{os.environ['MODEL']}/{os.environ['DOMAIN']}/{os.environ['BUNDLE']}/run_1"
fs = sorted(glob.glob(d + '/*.json'))
bad = passes = 0
for f in fs:
    r = json.load(open(f))['runs'][0]
    vs = r.get('verification_summary') or {}
    if r.get('error'):
        bad += 1; print('ERROR', os.path.basename(f)[:60], str(r['error'])[:90])
    if vs.get('total', 0) == 0:
        bad += 1; print('ZERO-VERIF', os.path.basename(f)[:60])
    passes += bool(r.get('overall_success'))
print(f'{len(fs)} files, {bad} suspect')
print(f'PASS {passes}/{len(fs)} = {100*passes/len(fs):.2f}%')
EOF
```

Export `MODEL`/`DOMAIN`/`BUNDLE` first, or inline the path. Any `error` value or
a `total: 0` verification summary means **stop and fix before trusting the run**.

To confirm the agent actually used tools (a trace can be present but inert),
count `tool_result` steps in `runs[0].conversation_flow`. Step objects carry
`type` ∈ {`system_message`, `user_message`, `ai_message`, `tool_result`} —
there is no `tool_call` step type, so counting that yields a misleading zero.

---

## 5. Re-running

`evaluate.py` **skips any task whose result file already exists.** To redo work,
delete the specific result files or use a fresh `--output_folder`. A run that
appears to finish instantly has skipped everything.

Check for a pre-existing directory *before* launching, or a partial earlier run
will quietly become your "complete" result set.

---

## 6. Before reporting numbers

Model configs default to `temperature: 0.6`, so a single run is one sample. The
same model has scored **89.2% and 85.1% on identical inputs** here. Differences
of a few points between models are not separable from noise.

Use `--num_runs 3`, or set `temperature: 0` in the config, before reporting
per-task verdicts or ranking closely-matched models.

---

## 7. Security

- Never print secret values. `source` env files with output suppressed and
  don't echo key variables — `${VAR:+set}` is safe, `${VAR:-…}` prints the
  value when set.
- Result files record context header **names** plus an allowlisted acting
  identity (e.g. `x-user-email`); credential-like header values are never
  written.
- Keep vendor bundles (`delivery_batch_*/`), generated configs
  (`.local_configs/`), results, and any report quoting task content **out of
  git**. Add each new bundle to `.gitignore`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Configuration must include either 'mcp_server_url' … or 'gym_servers_config'` | Vendor JSON fed straight to `evaluate.py` | Run `prepare_configs.py` first |
| `ModuleNotFoundError: langchain_openai` | Wrong interpreter | Use `.venv/bin/python` |
| Run "succeeds" with 0 result files | Per-task exceptions swallowed | Check file count, not exit code |
| Finishes instantly, files already there | Existing results skipped | Delete them or use a fresh output folder |
| `Gym '<domain>' not found in gym_servers_config` / `division by zero` | Verifier gym name ≠ MCP server name | Regenerate configs with `prepare_configs.py` |
| Every task fails on tool calls | Wrong host port, or server not running | Re-check `domain_conf.json`; `curl` the health endpoint |
| `LLMConfig.__init__() got an unexpected keyword argument` | Extra keys in the LLM config | Handled by `load_llm_configs`; update the repo |
| `Inference profile not found` (Bedrock) | Region not reaching the client | `aws_region` is aliased to `llm_region`; update the repo |
| `NoCredentialsError` (Bedrock) | Config credentials not forwarded to boto3 | Set `llm_api_key` + `llm_aws_secret_key` |
| Audit rejects traces for a foreign domain | `--domain` differed between the two steps | Keep `prepare_configs.py --domain` and the audit's `--domain` identical |

---

## Worked example

HR bundle, Luna with high thinking, 180 tasks — the run that produced
128/180 (71.11%):

```bash
docker run -d --name sn-hr-internal -p 8008:8005 sn-hr:latest
curl -s http://localhost:8008/health

.venv/bin/python prepare_configs.py \
  --input_dir  delivery_batch_09-30 \
  --output_dir .local_configs/hr_delivery_batch_09-30 \
  --mode oracle --domain hr --clean

.venv/bin/python evaluate.py \
  --configs_folder .local_configs/hr_delivery_batch_09-30 \
  --llm_config conf/llm/gpt_5p6_luna_high.json \
  --output_folder results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30 \
  --orchestrator react --concurrency 4 --num_runs 1

ls results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30/run_1/*.json | wc -l   # 180
```
