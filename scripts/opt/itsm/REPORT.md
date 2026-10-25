# ITSM prompt optimization: report

- **Model:** `conf/llm/gpt_5p6_terra_medium.json`, react orchestrator, concurrency 4.
- **Data:** `final_tasks/ITSM_181_Final` (vendor-refined bundle, 181 tasks). The configs were rebuilt fresh with `prepare_configs.py --mode oracle --clean` into `.local_configs/itsm_181_final` (0 skipped); no older `.local_configs` were used.
- **Split:** `scripts/opt/itsm/split.json`, **53 train / 128 test**. It was re-made at the user's request so test has at least 125 tasks: `make_split.py itsm 0.29 7`, stratified by task creator.
- **Budget:** 8 of the allowed 10 optimization steps were used (v1–v8). Repeat runs (`_r2`) measure noise and are not counted as steps. The user halted the run during a third v4 repeat; that partial run is excluded.

## Headline (two-run means)

| | train (53) | test (128) | all (181) |
|---|---|---|---|
| baseline (no overlay) | 66.0% | 60.9% | 62.4% |
| **v4 (selected, best test mean)** | 74.5% | **72.3%** | **72.9%** |
| v5 (runner-up, best train mean) | **79.2%** | 70.3% | **72.9%** |
| v3 | 78.3% | 70.7% | **72.9%** |
| **gain of v4 over baseline** | +8.5 | **+11.4** | **+10.5** |

- **Best single runs:** `step04_v4_r2` and `step05_v5_r2`, each 134/181 = 74.0%. `step04_v4_r2` also has the best test score, 94/128 = 73.4%.
- **Noise:** one overlay varies by about ±2–3 points between runs, and the baseline's test score moved 6 points between its two runs. The differences between v3, v4 and v5 are within that noise; all three average exactly 72.9% overall.
- **Selection:** v4 is chosen on the test (validation) mean, as the procedure specifies.
- **Alternative:** v5 drops two v4 clauses that diagnosis found harmful on train. It is the cleaner choice if you prefer train-side evidence, but it is not distinguishable from v4 on test.

## Attempts

| attempt | idea | train | test | all | verdict |
|---|---|---|---|---|---|
| baseline / _r2 | no overlay | 64.2 / 67.9 | 64.1 / 57.8 | 64.1 / 60.8 | reference |
| step01_v1 / _r2 | 9 principles added after the policy: duties not just restrictions, side effects, policy defaults, minimal fields, policy over user, transform rather than refuse, wider searches | 71.7 / 69.8 | 66.4 / 65.6 | 68.0 / 66.9 | +5 on test; the "think" clauses never ran (≈0 reasoning tokens per turn) |
| step02_v2 / _r2 | **visible procedure**: PLAN, a WRITE CHECK with each write, and a FINAL AUDIT in a reply that still calls a tool | 73.6 / 75.5 | 69.5 / 67.2 | 70.7 / 69.6 | +2.4 on test; structure is the main lever; lost v1's defaults, derived-value and transform clauses |
| step03_v3 / _r2 | PLAN quotes the policy sentences it relies on; per-parameter check on creates; policy defaults on create, update and messages (but not the tool's own defaults); derived and related values; facts vs free text; transform rather than refuse; word-by-word counting; Follow-up after each write; audit calls unused read tools | 79.2 / 77.4 | 70.3 / 71.1 | 72.9 / 72.9 | +2.3 on test; best stable train result |
| step04_v4 / _r2 | v3 plus: quote every branch of a rule; End state block and write ordering; check both linked records; search grid; same-meaning enum mapping; "unset needs a policy search"; counting always; audit fixes end-state mismatches | 73.6 / 75.5 | 71.1 / 73.4 | 71.8 / 74.0 | **best test mean (72.3)**; train −3.8 (enum and unset clauses hurt, plus dilution) |
| step05_v5 / _r2 | v3 plus only v4's net-positive additions | 81.1 / 77.4 | 68.0 / 72.7 | 71.8 / 74.0 | best train mean (79.2); test in the same band |
| step06_v6 | v5 with the policy moved into the user turn (`policy_to_user`) | 75.5 | 71.1 | 72.4 | no structural gain; not repeated |
| step07_v7 | v5 plus counting in 10-character segments, and keeping a named priority level as is | 77.4 | 69.5 | 71.8 | fixed both train truncation tasks; no test gain |
| step08_v8 | v7 condensed to about 3.1k characters | 73.6 | 68.0 | 69.6 | worse: the detail matters |

Full per-run notes are in `scripts/opt/itsm/attempts/LEDGER.md`.

## What worked
1. **Visible checks instead of "think about X" instructions** (v1 → v2). At medium effort the model spent about 0–40 reasoning tokens per turn and wrote no text between tool calls, so instructions to "re-read the policy" or "check unused tools" never ran. Making it write a PLAN, a per-write `field = value <- source` line and a FINAL AUDIT, all in replies that also call a tool, worked. A reply without a tool call ends the episode, so the audit has to sit in a tool-calling reply. The blocks were written in 53/53 train tasks, with no early endings.
2. **Quoting the policy, not ticking categories** (v3). With a category checklist the model wrote "SLA: none". Requiring it to quote the sentence it relied on surfaced duties buried mid-paragraph: SLA stand-down on reassignment, cascades on status changes, and link fields.
3. **Defaults as a separate, explicit rule**, covering updates and outgoing messages as well as creates, and **excluding defaults that only the tool describes**. Writing the tool's fallback value as if it were a policy default caused regressions in v2.
4. **Transform rather than refuse.** Format, prefix and length rules describe how to write a value, not whether to write it.
5. **End state and write ordering, quoting every branch of a rule, and a wider search grid** (v4/v5). Each fixed tasks that had failed in every earlier run.
6. **Counting in 10-character segments** (v7). It fixed both truncation tasks on train, where word-level tallies were off by 1–3 characters.

## What didn't
- **Picking an allowed value "by meaning"** (v4): it mapped "high … criticality" to `critical`. Requiring a policy search before leaving a field unset (v4) made the model discount facts the user stated.
- **"Restating the user's story is not a source"** (v2) dropped structured fields the user had supplied, such as the channel a request arrived by. Fixed in v3 by separating facts from free text.
- **"Write nothing extra"** (v1) dropped reference fields that the policy only implies. Fixed by the related-values clause.
- **Moving the policy into the user turn** (v6) gained nothing. **Condensing** the overlay (v8) lost about 3 points.
- **Self-audit by itself** confirmed the model's own plan instead of checking it against the policy. It helped only once it had to compare against the policy text and call tools.

## Remaining failures and suspected benchmark defects (train only)
Train tasks that still fail in most of the 14 overlay runs:
- `task_20260114_073549_564_391f1eff_18cc7ee8`: **partial defect.** KB matching is now right, but the worknote verifier rests on a conditional rule ("When worknotes record…").
- `task_20251222_190928_391_391f1eff_8cb006f5`: **defect.** The verifier requires a trailing period, but the policy's template sentence is ambiguous on that point. The golden run also fails.
- `task_20260116_235649_372_66392a82_ccb66a9a`: an intentional user-vs-policy test with indirect wording ("the opening entry" of the status list; looked-up tickets "keep" their fields). A fix would need wording close to this one task's policy, so it was left alone.
- `task_20260113_155812_591_82a7fa88_44904b62`: **verifier off-by-one.** It expects 224 characters for a 225-character limit, and runs that counted correctly failed.
- `task_20251218_193035_592_391f1eff_5293f595`: the expected phone value (stripping "91" and keeping the existing +1) is a contestable reading of the policy.
- `task_20260112_140854_400_c4ad7a70_63dff8c3`: a duty triggered by an event in the request (the caller's other on-hold ticket), plus the habit of adding a description.
- General: **most golden trajectories fail their own verifiers.** The tasks look hardened after the goldens were recorded. In almost every case the verifier traces to a policy sentence, so the goldens are out of date rather than the verifiers being wrong. The goldens were not treated as ground truth.

## Leakage audits
Each overlay was audited before its run by a subagent (v6 by a direct grep). The audit grepped distinctive phrases across all 181 configs (policy, request and verifiers), and v3–v8 also got a 4-gram scan. Findings that were rewritten:
- **v1:** the length clause closely paraphrased a policy sentence found in 4–5 tasks and encoded the verifier's outcome. It was made generic.
- **v2:** four near-paraphrases, each appearing in 4 tasks or fewer: "spaces included", "carries a value", a quoted "refers to", and "text templates". All were made generic.
- **v4:** "set the reference field on each side" was close to an idea found in only 3 train tasks. It was softened.
- **v3, v5, v7, v8:** passed. v7's level-mapping clause shares no distinctive words with the task that prompted it, but it is noted as an overfitting risk.

No overlay contains entity names, IDs, record numbers, dates or literal values. Test traces, verifiers and dumps were never opened; only aggregate test scores were used. No other domain's work was read. Full notes are in `LEDGER.md` under "Leakage audits".

## Artifacts
All paths are relative to the worktree `.claude/worktrees/itsm-prompt-opt` (branch `prompt-opt/itsm-terra`).

| What | Path | Tracked in git? |
|---|---|---|
| Selected overlay | `scripts/opt/itsm/overlays/v4.json` | yes |
| Runner-up overlay | `scripts/opt/itsm/overlays/v5.json` | yes |
| All overlays | `scripts/opt/itsm/overlays/v1.json … v8.json` | yes |
| Run ledger and audit log | `scripts/opt/itsm/attempts/LEDGER.md` | yes |
| Per-attempt overlay copy and scores | `scripts/opt/itsm/attempts/<attempt>/{overlay.json,scores.txt}` | yes |
| Diagnosis notes | `scripts/opt/itsm/attempts/{baseline,step01_v1,step02_v2,step03_v3,step04_v4}/analysis.md` | yes |
| Split | `scripts/opt/itsm/split.json` | yes |
| Raw traces per run | `results/opt/itsm/<attempt>/run_1/` | no (git-ignored) |
| Train failure dumps | `results/opt/itsm/<attempt>/failure_dumps/` | no (git-ignored) |
| Eval logs | `logs/opt_itsm/<attempt>.log` | no (git-ignored) |
| Configs | `.local_configs/itsm_181_final/oracle__itsm__<task_id>.json` | no (git-ignored); rebuild as below |
| Aborted partial run (excluded) | `results/opt/itsm/step04_v4_r3_ABORTED/` | no (git-ignored) |

## Reproduce
From the worktree root, with the ITSM MCP server up on `localhost:8006`:
```bash
.venv/bin/python prepare_configs.py --input_dir final_tasks/ITSM_181_Final \
  --output_dir .local_configs/itsm_181_final --mode oracle --domain itsm --clean
scripts/opt/run_attempt.sh itsm repro_v4 scripts/opt/itsm/overlays/v4.json
.venv/bin/python scripts/opt/compare.py itsm all baseline baseline_r2 -- repro_v4
```
To use the overlay outside this tooling, set `PROMPT_OVERLAY=<abs path to v4.json>` with the react orchestrator.
