# CSM_185tasks — Audit of tasks failed by both models

**Bundle:** `CSM_185tasks/` (185 tasks, seed `40945afa…_csm-01`)
**Models:** `gpt_6p1_sol`, `claude_opus_5_high` (react orchestrator, 1 run each, 2026-10-05)
**Results:** `results/react/<model>/csm/csm_185tasks/run_1/`
**Scope:** the 15 tasks that **both** models failed. Each was analysed against the vendor task (prompt, policy, verifier SQL), the DB seed, and both traces; the key claim of every analysis was then spot-checked against the files.

## Headline

| | Count |
|---|---|
| Benchmark gap | **11** |
| Agent issue | **3** |
| Mixed (one model each way) | **1** |

**About 11 of the 15 tasks both models failed are caused by the benchmark, not by model weakness.**

### Pass/fail matrix (all 185)

| | claude pass | claude fail | total |
|---|---|---|---|
| **gpt pass** | 135 | 19 | 154 (83.2%) |
| **gpt fail** | 16 | **15** | 31 |
| **total** | 151 (81.6%) | 34 | 185 |

### What fixing the gaps would change

If the 11 gaps were fixed so that the models' (reasonable) answers were accepted:

- `gpt_6p1_sol`: 154 → **165 / 185 (89.2%)**
- `claude_opus_5_high`: 151 → **162 / 185 (87.6%)**

These projections cover only the both-fail set. The 35 tasks that just one model failed were not audited, and at least 3 of them carry the "third variant" pattern (see below).

---

## Per-task verdicts

Confidence: H = high, MH = medium-high, M = medium.

| # | Task (suffix) | Failing check | gpt | claude | Overall | Root cause |
|---|---|---|---|---|---|---|
| 1 | `8c7a6205` | assignment group (2 checks) | BENCH (MH) | BENCH (MH) | **BENCH (MH)** | Verifier wants `Escalation Management_3`; prompt says "Escalation Management"; only hint is garbled "third variant" |
| 2 | `a86667c0` | assignment group (2 checks) | BENCH (MH) | BENCH (MH) | **BENCH (MH)** | Same pattern. Verifier wants `Case Assignment_3`; the earlier revised version named it in the prompt |
| 3 | `c124578d` | escalation_reason | BENCH (MH) | BENCH (MH) | **BENCH (MH)** | Prompt says "high urgency"; verifier wants `customer_request` (policy "by default") |
| 4 | `47b48a98` | phone number | BENCH (M) | BENCH (M) | **BENCH (M)** | Ambiguous "dialing prefix" rule; verifier wants an invented `+44` number, models refused |
| 5 | `ded1413d` | case update / group | AGENT (M) | AGENT (H) | **AGENT (MH)** | Claude never updated the existing case; GPT left the group off the new case |
| 6 | `b7edfe7c` | KB choice / short_description | BENCH (M) | BENCH (M)\* | **BENCH (M)** | KB 125 vs 126 decided by a garbled "config drift" hint; exact match on trailing period |
| 7 | `6abc29d7` | user deactivated | AGENT (M) | AGENT (M) | **AGENT (M)** | "Misguided user" trap: both noticed Joanne is a manager, not an agent, and deactivated her anyway |
| 8 | `7ba653d0` | KB publish + link (+ GPT assignment) | MIXED (M) | BENCH (M) | **BENCH (M)** | Verifier expects an unrequested re-publish of a retired KB; GPT also stopped early |
| 9 | `1a268d87` | interaction channel | BENCH (H) | BENCH (H) | **BENCH (H)** | Verifier wants the old interaction relabelled `email` because the user "mailed"; vendor note admits this |
| 10 | `f075229a` | interaction contact_id | AGENT (M) | AGENT (M) | **AGENT (M)** | Both left `contact_id` off the interaction, although the emailer is the case contact (124) |
| 11 | `a933399b` | warranty_end | BENCH (H) | BENCH (H) | **BENCH (H)** | Verifier wants the replacement to inherit the old warranty; vendor note: "SM does not say inherit" |
| 12 | `7304e48e` | warranty_end | BENCH (H) | BENCH (H) | **BENCH (H)** | Verifier wants warranty = contract end; the prompt never mentions warranty |
| 13 | `187f2808` | escalation_reason | BENCH (H) | BENCH (H) | **BENCH (H)** | Prompt says "significant impact on the business"; verifier wants `customer_request` |
| 14 | `4dd068cf` | short_description / state / membership | BENCH (H) | AGENT (MH) | **MIXED (M)** | GPT failed only on a trailing period; Claude skipped group membership and moved the case to in_progress unasked |
| 15 | `557797b1` | interaction contact_id | BENCH (M) | BENCH (H) | **BENCH (M)** | Verifier wants contact 123 (Elizabeth Chase), but the person writing in is Jose, who has no contact record |

\* **Override:** the analyst rated Claude's dropped trailing period in #6 as an agent issue (M). I reclassified it as a benchmark gap so it matches #14, where the opposite happened (the verifier *rejects* the period that GPT kept). The bundle is inconsistent about trailing punctuation, so neither model should be penalised for it. With the period fixed, Claude passes #6, because it linked the correct KB (126).

**Per model** (15 tasks):

| Model | Benchmark | Agent | Mixed |
|---|---|---|---|
| `gpt_6p1_sol` | 11 | 3 (#5, #7, #10) | 1 (#8) |
| `claude_opus_5_high` | 11 | 4 (#5, #7, #10, #14) | 0 |

**Confidence of the overall verdicts:**

| Confidence | Count | Tasks |
|---|---|---|
| High | 4 | all benchmark gaps |
| Medium-high | 4 | 3 benchmark gaps, 1 agent issue |
| Medium | 7 | 4 benchmark gaps, 2 agent issues, 1 mixed |

---

## Benchmark gap patterns

| Pattern | Tasks | Fix |
|---|---|---|
| **Requirement hidden in a garbled policy fragment** | #1, #2 ("third variant" → `_3` group), #6 ("config drift" → KB 126) | Put the target in the prompt, or write a clear policy rule |
| **Expected value never stated (the verifier assumes an inference)** | #12 (warranty = contract end), #11 (inherit warranty), #9 (mailed → email channel), #8 (re-publish a retired KB) | Say it in the prompt or policy, or drop the condition |
| **Enum value contradicts the prompt's wording** | #3 (urgency), #13 (impact) vs `customer_request` | Accept several values, or remove the conflicting wording |
| **Ambiguous policy rule** | #4 (phone "dialing prefix") | Spell out the rewrite, or expect the number unchanged |
| **Verifier expects the wrong entity** | #15 (another person's contact) | Drop or relax the `contact_id` condition |
| **Brittle exact string match** | #14 (GPT), #6 (Claude): trailing period, required in one task and rejected in the other | Compare with `RTRIM(..., '.')`, or quote descriptions in the prompt |

**Vendor awareness:** in #9 and #11 the vendor's own `scenario_description` admits that the policy doesn't support the expected value ("SM does not say mailed means email", "SM does not say inherit"). The vendor's reference runs (claude-opus-4.5) also scored pass@5 = 0 on #8, #9 and #12.

### Bundle-wide spread of the recurring patterns

| Pattern | Tasks containing it | Both fail | One model fails | Both pass |
|---|---|---|---|---|
| "second/third/fourth variant" group hint | 6 | 2 | 3 (`910ff140`, `e3b2012c`, `0dce7d7e`) | 1 |
| "dialing prefix" rule | 3 | 2 (`47b48a98`, `f075229a`\*\*) | 0 | 1 |
| "By default escalation reason" | 1 | 1 | 0 | 0 |

\*\* `f075229a` carries the rule, but it failed for an unrelated reason (agent issue #10).

---

## Agent issues

| Task | Model | What went wrong | Confidence |
|---|---|---|---|
| #5 `ded1413d` | claude | Found existing case 50 but never escalated or assigned it; applied everything to the new case only | H |
| #5 `ded1413d` | gpt | Created the copied case without `assignment_group_id`, reading the policy's copy rule too literally | M |
| #7 `6abc29d7` | both | Noticed Joanne is a manager, not the "agent" named, but deactivated her anyway; the policy says not to act on clearly misguided requests | M |
| #10 `f075229a` | both | Logged the email interaction without the case's `contact_id` 124, although it was in the case record and the tool accepts it. Claude also wrongly stated that Pamela isn't the contact | M |
| #14 `4dd068cf` | claude | Assigned Stephanie without adding her to the group, and moved the case to `in_progress` unasked | MH |
| #8 `7ba653d0` | gpt (part) | Stopped all work when the KB step was blocked, instead of doing the assignment and SLA steps it could still do | M |

#7 and #10 also have minor wording gaps (whether a wrong role makes the user "clearly misguided"; no explicit rule for an interaction's contact), but the failures are mainly the agents'.

---

## Full task IDs

| # | Task ID |
|---|---|
| 1 | `task_20251205_153330_906_a8eea1c0_8c7a6205` |
| 2 | `task_20260104_202644_765_00364ece_a86667c0` |
| 3 | `task_20251207_174704_642_5c0cd3a2_c124578d` |
| 4 | `task_20251210_113549_827_d4a463c3_47b48a98` |
| 5 | `task_20251204_160436_303_c4ad7a70_ded1413d` |
| 6 | `task_20251126_211806_641_391f1eff_b7edfe7c` |
| 7 | `task_20251201_173129_319_2aa6b0ad_6abc29d7` |
| 8 | `task_20251205_075543_102_46ca2862_7ba653d0` |
| 9 | `task_20251206_113950_467_6cd2e092_1a268d87` |
| 10 | `task_20251210_002938_761_897c3e0b_f075229a` |
| 11 | `task_20251210_201233_463_c4ad7a70_a933399b` |
| 12 | `task_20260101_043715_660_a8eea1c0_7304e48e` |
| 13 | `task_20260101_233731_695_00364ece_187f2808` |
| 14 | `task_20260102_183520_960_e6427839_4dd068cf` |
| 15 | `task_20260104_161718_180_46ca2862_557797b1` |

---

## Suggested next steps

1. Return the 11 gap tasks to the vendor with the fixes above. Prioritise the 4 high-confidence ones (#9, #11, #12, #13) and the "third variant" pattern, which affects 6 tasks.
2. Audit the 35 tasks that just one model failed. Start with the 3 that contain "third variant".
3. Apply a bundle-wide fix for trailing-punctuation matching.
4. Re-score after the fixes before comparing the two models; the current 1.6-point gap is within the noise.
