# Running Evaluations

How to evaluate the **csm**, **itsm** and **hr** task bundles with any model in
`conf/llm/`. This covers evaluation only — producing traces and a pass rate. To
classify failures as benchmark defects vs. agent mistakes, continue with
[`SAMPLE_AUDIT.md`](SAMPLE_AUDIT.md).

```
.local_configs/<domain>_<bundle>  ──►  evaluate.py  ──►  traces + pass rate
   (already prepared)                                     one file per task
```

Budget ~20–45 min per model for ~180 tasks at `--concurrency 4`.

---

## 1. The three domains

Configs are **already prepared** for all three — you do not need to run
`prepare_configs.py` unless a new bundle arrives (see §6).

| Domain | Config folder | Tasks | MCP server | Port | Image tar |
|---|---|---:|---|---:|---|
| `csm` | `.local_configs/csm_185tasks` | 185 | `sn-csm-server` | 8001 | `sn-csm.tar` |
| `itsm` | `.local_configs/itsm_delivery_batch_09-24` | 165 | `gym-itsm-mcp` | 8006 | `sn-itsm.tar` |
| `hr` | `.local_configs/hr_delivery_batch_09-30` | 180 | `sn-hr-internal` | 8008 | `sn-hr.tar` |

The server names and ports come from `conf/ray/domain_conf.json` and are baked
into every config as `mcp_server_name` / `mcp_server_url`. **The ports are not
contiguous** — csm 8001, itsm 8006, hr 8008. Don't infer them.

Tars live in the repo root, next to `evaluate.py`.

### Available models

`ls conf/llm/` — `claude_opus_5`, `claude_opus_5_high`, `gpt_5p6_luna`,
`gpt_5p6_luna_high`, `gpt_5p6_sol`, `gpt_5p6_terra`, `gpt_5p6_terra_high`,
`gpt_6p1_sol`. The `_high` suffix means high thinking effort.
(`my-model.json` and `openrouter.json` are templates, not real models.)

Model configs may contain credentials. Don't read or print them; pass the path.

---

## 2. Bring the MCP server online (if it isn't already)

**Check first — all three are often already running:**

```bash
curl -s http://localhost:8001/health    # csm  → {"status":"healthy","service":"sn-csm"}
curl -s http://localhost:8006/health    # itsm → {"status":"healthy","service":"itsm-api"}
curl -s http://localhost:8008/health    # hr   → {"status":"healthy","service":"hr-api"}
```

If a port answers `healthy`, skip to §3. Otherwise load the tar and start it:

```bash
# csm
docker load -i sn-csm.tar  && docker run -d --name sn-csm-server  -p 8001:8005 sn-csm:latest
# itsm
docker load -i sn-itsm.tar && docker run -d --name sn-itsm-server -p 8006:8005 sn-itsm:latest
# hr
docker load -i sn-hr.tar   && docker run -d --name sn-hr-internal -p 8008:8005 sn-hr:latest
```

Every container listens internally on **8005**; only the host-side port differs.
The container *name* is arbitrary — the port binding is what matters, since that
is what the configs point at.

`docker load` is idempotent; if the image is already present it's a no-op. If
the container name is taken but stopped, `docker start <name>` rather than
`run`. To check what's up:

```bash
docker ps
```

> **Don't confuse these with the V2 containers.** A separate
> `EnterpriseOps-Gym-V2` stack may be running `mcp_server-mcp-hr-1` on **8017**
> and `mcp_server-mcp-itsm-1` on **8016** from different images. Those are *not*
> the servers these configs target. Health-check the port from the table, not
> whichever hr/itsm container happens to be up.

On Apple Silicon an `amd64` image runs under emulation — functional, slower.

---

## 3. Evaluate

Pick a domain's config folder from §1 and a model, then:

```bash
.venv/bin/python evaluate.py \
  --configs_folder .local_configs/hr_delivery_batch_09-30 \
  --llm_config conf/llm/gpt_5p6_luna_high.json \
  --output_folder results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30 \
  --orchestrator react --concurrency 4 --num_runs 1
```

The `--output_folder` convention is
`results/react/<model>/<domain>/<bundle>`; traces land in
`…/run_1/results_oracle__<domain>__<task_id>.json`. That filename pattern is
what `auto-detect` requires, so keep the convention if you plan to audit.

Parameterized over all three domains:

```bash
MODEL=gpt_5p6_luna_high

run_domain() {   # $1=domain  $2=config folder  $3=bundle label
  .venv/bin/python evaluate.py \
    --configs_folder ".local_configs/$2" \
    --llm_config "conf/llm/${MODEL}.json" \
    --output_folder "results/react/${MODEL}/$1/$3" \
    --orchestrator react --concurrency 4 --num_runs 1
}

run_domain csm  csm_185tasks             csm_185tasks
run_domain itsm itsm_delivery_batch_09-24 delivery_batch_09-24
run_domain hr   hr_delivery_batch_09-30   delivery_batch_09-30
```

**Use `.venv/bin/python`, never system `python3`.** The system interpreter may
lack `langchain_openai` / `langchain_anthropic`; every task then fails with
`ModuleNotFoundError` while `evaluate.py` still exits 0.

**Run one model at a time**, and one domain at a time. Models share the MCP
container; concurrent runs double the database create/seed/delete load and
produce spurious failures.

Other orchestrators (`planner_react`, `decomposing`) additionally require
`--planner_llm_config`. `--hf_dataset` replaces `--configs_folder` for pulling
tasks from HuggingFace, and is the only case where `evaluate.py`'s own
`--domain`/`--mode` flags do anything.

---

## 4. Verify the run — the exit code is not a success signal

`evaluate.py` swallows per-task exceptions and **exits 0 even if every task
failed**. Check the file count against the task count from §1:

```bash
ls results/react/$MODEL/hr/delivery_batch_09-30/run_1/*.json | wc -l    # expect 180
```

Then confirm the traces are real, not merely present:

```bash
.venv/bin/python -I - <<'EOF'
import json, glob, os, sys
d = 'results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30/run_1'
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

Any `error` value or a `total: 0` verification summary means **stop and fix
before trusting the run**.

To confirm the agent actually used tools (a trace can be present but inert),
count `tool_result` steps in `runs[0].conversation_flow`. Step objects carry
`type` ∈ {`system_message`, `user_message`, `ai_message`, `tool_result`} —
there is **no `tool_call` step type**, so counting that yields a misleading zero.

---

## 5. Re-running

`evaluate.py` **skips any task whose result file already exists.** To redo work,
delete the specific result files or use a fresh `--output_folder`. A run that
finishes suspiciously fast has probably skipped everything — check for a
pre-existing output directory *before* launching, or a partial earlier run
quietly becomes your "complete" result set.

---

## 6. Only if a new bundle arrives

The three folders in §1 are already built. For a newly delivered bundle:

```bash
.venv/bin/python prepare_configs.py \
  --input_dir  delivery_batch_<new> \
  --output_dir .local_configs/<domain>_delivery_batch_<new> \
  --mode oracle --domain <domain> --clean
```

Vendor JSON is **not** what `evaluate.py` reads — it nests fields under
`config`, uses `gym_server_id` instead of a server name/URL,
`verifier_configs` instead of `verifiers`, and carries no seed path. This step
resolves the MCP name/URL from `domain_conf.json`, attaches an absolute
`seed_database_file` path, reshapes verifiers, and remaps each verifier's
target gym to the server name.

**Read the last line** — skipped tasks mean the eval silently covers fewer tasks
than were delivered:

```
Wrote 180 configs to .local_configs/hr_delivery_batch_09-30 (0 skipped)
```

`--mode` and `--domain` are **only labels used in output filenames**. `--mode
oracle` does *not* mean oracle/replay — these are real LLM runs. The labels
still matter: they propagate into result filenames, and `auto-detect` rejects
any trace whose filename lacks the `__<domain>__` segment it was invoked with.
`--seed_dir` defaults to `<input_dir>/db_seeds`.

Add the new bundle directory to `.gitignore`.

---

## 7. Before reporting numbers

Model configs default to `temperature: 0.6`, so a single run is one sample. The
same model has scored **89.2% and 85.1% on identical inputs** here. Differences
of a few points between models are not separable from noise.

Use `--num_runs 3`, or set `temperature: 0` in the config, before reporting
per-task verdicts or ranking closely-matched models.

---

## 8. Security

- Never print secret values. `source` env files with output suppressed and
  don't echo key variables — `${VAR:+set}` is safe, `${VAR:-…}` prints the
  value when set.
- Result files record context header **names** plus an allowlisted acting
  identity (e.g. `x-user-email`); credential-like header values are never
  written.
- Keep vendor bundles, `.local_configs/`, results, and any report quoting task
  content **out of git**.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_openai` | Wrong interpreter | Use `.venv/bin/python` |
| Run "succeeds" with 0 result files | Per-task exceptions swallowed | Check file count, not exit code |
| Finishes instantly, files already there | Existing results skipped | Delete them or use a fresh output folder |
| Every task fails on tool calls | Server down, or wrong port | `curl` the port from §1; start the tar |
| Health check passes but tasks still fail | Health-checked a V2 container on 8016/8017 | Use the §1 port for that domain |
| `Configuration must include either 'mcp_server_url' … or 'gym_servers_config'` | Vendor JSON fed straight to `evaluate.py` | Run `prepare_configs.py` (§6) |
| `Gym '<domain>' not found in gym_servers_config` / `division by zero` | Verifier gym name ≠ MCP server name | Regenerate configs (§6) |
| `LLMConfig.__init__() got an unexpected keyword argument` | Extra keys in the LLM config | Handled by `load_llm_configs`; update the repo |
| `Inference profile not found` (Bedrock) | Region not reaching the client | `aws_region` is aliased to `llm_region`; update the repo |
| `NoCredentialsError` (Bedrock) | Config credentials not forwarded to boto3 | Set `llm_api_key` + `llm_aws_secret_key` |
| Audit rejects traces for a foreign domain | `--domain` differed between prepare and audit | Keep them identical |

---

## Worked example

HR, Luna with high thinking — the run that produced 128/180 (71.11%):

```bash
curl -s http://localhost:8008/health        # already healthy; no docker needed

.venv/bin/python evaluate.py \
  --configs_folder .local_configs/hr_delivery_batch_09-30 \
  --llm_config conf/llm/gpt_5p6_luna_high.json \
  --output_folder results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30 \
  --orchestrator react --concurrency 4 --num_runs 1

ls results/react/gpt_5p6_luna_high/hr/delivery_batch_09-30/run_1/*.json | wc -l   # 180
```

HR pass rates measured this way (single run, `temperature: 0.6`, 180 tasks):
gpt-6.1-sol 87.78%, claude-opus-5-high 86.67%, gpt-5.6-terra 74.44%,
gpt-5.6-terra-high 73.89%, gpt-5.6-luna-high 71.11%, gpt-5.6-luna 59.44%.
