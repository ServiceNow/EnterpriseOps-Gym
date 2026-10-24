# step01_v1 train analysis

## Regressions (passed in a baseline run, failed under v1)
- ..._cadddb04 — CAUSED by v1 clause 4 ("write nothing extra"): dropped `service`/`service_offering`/`original_task` that the policy implies via a descriptive statement (value lives on the related open ticket). Trade-off: clause 4 also fixed baseline_r2's extra short_description.
- ..._9c6f3e01 — noise: digit-slicing slip (baseline 1/2 also failed). Suggest: strip separators, count digits one by one, then format.
- ..._fad8a09d — CAUSED by clause 4 + call order: created problem before listing related incidents, left `original_task` unset (policy states it descriptively: "X has Y"), never back-filled.
- ..._02a35b2b — noise (baseline 1/2): user said no KB article; policy duty to document resolution. v1 "policy outranks user" didn't override an explicit user "don't".

Proposed: clause 4 → "every argument needs a source (request, policy default, or value the policy ties to a looked-up related record); descriptive policy statements about what a record has/belongs to are instructions to write that field". Add: look up related records before the first create, fill reference fields in the same call; back-fill if created without. Add: explicit user "don't do X" does not cancel a policy-imposed duty.

## Persistent failures, part A (5 tasks)
KEY: model emits no visible text except the final answer and ~0–40 reasoning tokens/turn; v1's "read/check/re-check" clauses are mental instructions that never execute. v1 helped where it changed a concrete habit (refusals, enum defaults, search by type, SLA on create), not via any review pass.
- dc6af68f — same: reassignment SLA stand-down (indirect idiom); 2 SLA tools unused.
- 8487a7f6 — changed (v1 fixed group-by-type + SLA link); new: assignee not in group → left whole assignment blank instead of setting group and leaving assignee empty (partial-block at field level).
- a76dbe57 — same: problem link on major incident never set; finished after 3 turns; list/update tools unused. Clearest case for a forced audit.
- 48850a98 — changed (v1 fixed resolution_code + exact text); new: notification type not set to the enum value matching user intent (clause 4 side effect). Low priority.
- 8cb006f5 — changed (v1 fixed refusal); new: missing follow-on note after KB link (chained duty after the last write). Golden also fails.
Proposals: visible PLAN block before first write (fixed record-type checklist), re-check triggers after each write, field-level partial blocks, AUDIT table of unused tools gating final answer; shorten overlay; optional user-turn trailer.

## Persistent failures, part B (6 tasks)
KEY (agrees with A): 0–196 reasoning tokens/turn, no visible text; a text-only message ends the episode (react.py), so "final re-check" can only occur in the last message, usually at 0 reasoning. Only cheap literal clauses took effect. Clause 8 (copy quoted text exactly) contradicts clause 6 (truncate) in both length-limit tasks.
- 63dff8c3 — same: extra description; misses rule triggered by a request event (not by own write).
- f89140aa / b09b6872 — same: 3/2 chars over cap, no counting. Fix: written running tally then cut.
- 18cc7ee8 — slightly worse: narrow text-filter KB search fired before planning. Worknote check relies on conditional rule → partly defect, deprioritise.
- 5febbc10 — same: cascade buried mid-sentence; 4/8 tools unused; unused-tool check never ran.
- ccb66a9a — same: user wording won over indirect policy wording (initial status, preserved fields). Intentional test.
Proposals: per-write visible "write check" (arg <- source), visible "triggers" block after lookups, final unused-tools check in a message that still calls a tool; shorten; try policy_to_user.
