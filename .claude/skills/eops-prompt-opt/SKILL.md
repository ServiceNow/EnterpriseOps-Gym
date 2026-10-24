---
name: eops-prompt-opt
description: Automated prompt optimization (hill climbing) for one EnterpriseOps-Gym domain (itsm, hr, csm) on the *_Final task bundles with gpt_5p6_terra_medium. It wraps task prompts with a meta-instruction overlay (task text untouched), diagnoses train failures with subagents, iterates for up to N optimization steps, and reports. Use when the user runs /eops-prompt-opt <domain> or asks to hill-climb / optimize prompts for an EOPS domain.
argument-hint: <itsm|hr|csm> [--steps 10] [--llm conf/llm/gpt_5p6_terra_medium.json]
---

# EOPS prompt optimization (hill climbing)

Arguments: `$ARGUMENTS` (first token = domain; default steps = 10).

The goal is to raise the target model's pass rate on the domain's benchmark by optimizing *meta-instructions*
around the task prompts, using a train split for diagnosis and a held-out test split as validation.

## Rules of the game (from the user; do not relax)
- **Never alter task-supplied text.** You may add meta-instructions: prepend or append to the system or user
  message, or move the system prompt into the user turn. The overlay hook passes task text through verbatim.
- **No cheating / leakage.** An overlay must not contain task-specific names, IDs, record numbers, values or
  dates, and must not closely paraphrase a policy sentence that only a handful of tasks contain. Target
  *families* of errors with reusable behaviours.
- **Train:** you may read golden trajectories, failing verifier SQL and policies.
  **Test:** use aggregate pass rates only (as validation). Never open test traces, verifiers or dumps.
  `compare.py` deliberately prints per-task flips for train only.
- **Budget:** at most `--steps` optimization steps; one step = one new overlay evaluated on all tasks.
  Repeat runs of an already-evaluated overlay, done to measure noise, are not steps. State this in the report.
- **Model:** `conf/llm/gpt_5p6_terra_medium.json` unless overridden. Never read or print model configs;
  they contain secrets. Pass the path only.
- **Independent exploration.** Other domains have been, or are being, optimized separately. Do **not** read
  their overlays, analyses, reports or results: other `prompt-opt/*` branches or worktrees, and
  `results/opt/` outside your domain. Derive every hypothesis from this domain's own data.

## 0. Setup (once per session)
1. Create a dedicated worktree from branch **`prompt-opt/tooling`**. It has the overlay hook, the client
   changes the terra configs need, the optimization scripts, and the pre-made splits. `main` lacks these.
   From the main checkout `<REPO>`:
   ```bash
   git worktree add .claude/worktrees/<dom>-prompt-opt -b prompt-opt/<dom>-terra prompt-opt/tooling
   ```
   then call EnterWorktree with `path=<REPO>/.claude/worktrees/<dom>-prompt-opt`. Inside it:
   ```bash
   R=<REPO>; ln -s $R/.venv .venv; ln -s $R/conf conf; ln -s $R/final_tasks final_tasks
   ```
2. Check the MCP server: `curl -s localhost:<port>/health`. Ports are in `scripts/opt/domains.py`
   (csm 8001, itsm 8006, hr 8008). If it's down, follow `docs/RUN_EVALS.md` §2.
3. Build configs (git-ignored, so rebuild in every new worktree):
   ```bash
   .venv/bin/python prepare_configs.py --input_dir <bundle from domains.py> \
     --output_dir <configs from domains.py> --mode oracle --domain <dom> --clean
   ```
   The last line must say `0 skipped`.
4. The split `scripts/opt/<dom>/split.json` is already committed (stratified by task-creator hash, seed 7).
   Keep it. `make_split.py <dom>` refuses to overwrite.

## 1. Tooling (domain-aware; run from the worktree root)
| Command | Purpose |
|---|---|
| `scripts/opt/run_attempt.sh <dom> <name> [overlay.json]` | full eval of all tasks (react, concurrency 4), then auto-archive. Run it with `run_in_background: true` and wait for the completion notification. |
| `.venv/bin/python scripts/opt/archive_attempt.py <dom> <name> <overlay\|none> [analysis.md]` | (re)archive scores and the prompt; attach analysis notes |
| `.venv/bin/python scripts/opt/compare.py <dom> <train\|test\|all> A... -- B...` | group mean pass rates; train-only flips (gained/lost) |
| `.venv/bin/python scripts/opt/dump_failures.py <dom> <name> [--prev A B ...]` | per-task train failure dumps (agent trajectory, golden trajectory, failing SQL) plus `lost.txt` (passed in a `--prev` attempt) and `rest.txt` |
| `.venv/bin/python scripts/opt/history.py <dom> [name]` | for each failing train task, how many attempts it failed in |
| `.venv/bin/python scripts/opt/make_variants.py <base.json> <prefix_out.json> <policy_to_user_out.json>` | derive structural variants of an overlay |
| `.venv/bin/python scripts/opt/score.py <run_dir> [split.json key]` | score one run |

**Overlay format** (JSON; every key optional): `system_prefix`, `system_suffix`, `user_prefix`, `user_suffix`,
`policy_to_user` (bool: the task system prompt moves into the user turn inside `<operating_policy>` tags,
and the request goes inside `<user_request>`). The hook is `apply_prompt_overlay` in
`orchestrators/react.py`, selected by `$PROMPT_OVERLAY`.

**Layout:**
- overlays: `scripts/opt/<dom>/overlays/vN.json`
- tracked per-attempt artifacts: `scripts/opt/<dom>/attempts/<name>/{overlay.json,scores.txt,analysis.md}`
- run ledger: `scripts/opt/<dom>/attempts/LEDGER.md`
- heavy traces, logs and dumps: `results/opt/<dom>/<name>/` (git-ignored)

Name attempts `baseline`, `baseline_r2`, `stepNN_vN`, `stepNN_vN_r2`.

## 2. Procedure
1. **Baseline:** run `baseline` and `baseline_r2` with no overlay. Two runs give you a noise estimate
   before you compare anything.
2. **Diagnose on train:** `dump_failures.py <dom> <attempt> [--prev ...]`. Launch 2–3 `general-purpose`
   subagents in parallel over disjoint subsets of the dumps (regressions in `lost.txt` get their own agent).
   Give each one:
   - absolute paths to the dumps, the current overlay, the configs (policy and verifiers), and earlier
     attempts' traces
   - the ask: root cause per task (compare against the failing verifier SQL and the task policy), a
     general failure family, which overlay clause should have caught it and why it didn't,
     protocol-caused vs sampling noise, suspected benchmark defects, and draft general wording with no
     task-specific content
   - a check of whether golden trajectories actually satisfy the current verifiers before treating them
     as ground truth

   Subagents cannot write files, so save their reports yourself into
   `scripts/opt/<dom>/attempts/<attempt>/analysis.md`.
3. **Write or revise the overlay** to address the largest fixable families. Form hypotheses from this
   domain's failures and policies. Consider both content (what to instruct) and structure (where it goes;
   system vs user turn; length).
4. **Leakage audit before every run:** a short subagent greps distinctive overlay phrases across all
   configs and verifiers and flags entity names, values, and paraphrases that occur in only a few tasks.
   Rewrite anything it flags. Record the audit in the ledger.
5. **Evaluate:** `run_attempt.sh <dom> stepNN_vN overlays/vN.json`. Compare with `compare.py`.
   Single runs are noisy, so repeat promising overlays (`_r2`) and decide on two-run means.
   **Select on the test (validation) mean**, and watch for train/test divergence.
6. After every run, update `LEDGER.md` and commit (`git add scripts/opt; git commit`), so the work survives
   worktree removal.

## 3. Operational gotchas (tooling, not findings)
- Don't wait with `pgrep -f "<pattern>"` loops. The loop's own command line matches the pattern and never
  exits. Use `run_in_background` and the completion notification.
- Don't edit a shell script while a background run is executing it; bash reads scripts incrementally.
  The hook also re-reads the overlay file per task, so don't modify or delete an overlay mid-run.
- Under worktree isolation, complex one-liners can be refused (the repo path contains "git"; variables
  feeding commands). Put logic in small scripts and call them plainly.
- Config filenames are `oracle__<dom>__<task_id>.json`.
- `evaluate.py` exits 0 even when tasks crash. `score.py` counts every split task, so missing traces show
  as `missing N`; investigate any nonzero value.
- One eval at a time per domain (each domain has its own MCP server). Sessions for different domains share
  the LLM endpoint, so runs may be slower.

## 4. Final report
Write `scripts/opt/<dom>/REPORT.md`. Include:
- a headline table (baseline vs best, two-run means, train/test/all)
- one row per attempt with its idea, scores and verdict
- what worked and what didn't
- residual failures and suspected benchmark defects
- leakage-audit notes
- artifact pointers
- a reproduce command

Commit it, then end with the `result:` line.
