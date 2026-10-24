# Baseline train failure analysis (19 failures / 53 train)

Three parallel subagents, disjoint task subsets. All failures judged protocol-caused (systematic) except one digit slip.
Most golden trajectories FAIL their own failing verifiers (hardened tasks / stale goldens); in nearly every case the
failing check is grounded in a (often buried, mid-sentence) policy clause.

## Families (approx. counts across 19 tasks)
1. **Missed policy-imposed duties (~9)** — side effects on related records (SLA start on create, SLA stand-down on
   reassignment, SLA pause on modification, cascades on state transitions, link field on the record a new record
   relates to), defaults for optional fields stated in policy prose (status / codes / companion status on update),
   policy-mandated note fields. Agent treats the user request as the checklist; reads policy only for blockers.
2. **Extra unrequested optional fields (3)** — description, subject, category, plans added "for completeness".
3. **Narrow lookups (3)** — single text-filter search on one field; name search instead of the attribute policy
   routes by; keyword search used as if exhaustive for a policy-defined matching set.
4. **Length / format rules (3)** — no character counting for length caps with truncation; format rule treated as a
   blocker instead of normalizing.
5. **Policy precedence (3)** — user wording overrode a policy precondition / fixed value (2); invented constraint
   from preamble wording → refused entire task (1).
6. **Literal / extraction slips (2)** — punctuation added to quoted text; digit slicing slip.
7. Prose-to-enum mapping slip (1) — "high ... criticality" read as `critical`.

Suspected benchmark issues: phone-prefix expected value contestable (task ..._5293f595); conditional worknotes rule
enforced as mandatory (..._18cc7ee8).

---
Per-batch reports follow.

## Batch A (6 tasks)

1. task_..._dc6af68f — reassignment; missed policy side effect: stand down open SLA on handoff. Systematic. Golden also fails.
2. task_..._8487a7f6 — (a) group lookup by name only, wrong org, left unassigned (should search by type); (b) missed SLA linking on create. Systematic. Golden fails.
3. task_..._be0bd794 — notification status default ("sent") buried in policy, omitted → queued. Systematic. Golden fails.
4. task_..._5293f595 — phone format/prefix rule read as blocker; dropped the write instead of normalizing. Golden fails; expected value contestable (suspected defect).
5. task_..._a76dbe57 — created problem but never linked major incident's problem field (explicit rule). Systematic. Golden fails.
6. task_..._48850a98 — (a) missed default resolution_code enum; (b) added trailing '.' to quoted notification message.

Cross-cutting: none of the 6 goldens pass their failing verifiers; agent treats user request as the checklist, reads policy only for blockers; unused exposed tools were exactly the ones needed in 3 tasks.

Ranked: missed policy side effects on related records 3; missed policy defaults 2; refused instead of normalizing 1; single-search lookup 1; literal string altered 1.

Clauses: policy duty sweep before finishing; policy defaults are mandatory (map to enum); unused tools as cues; normalize, don't refuse; robust lookups + exact strings.
## Batch B (6 tasks)

1. task_..._8cb006f5 — invented constraint (caller/CI org mismatch) → refused whole task. Family: over-cautious refusal. Systematic.
2. task_..._9c6f3e01 — phone digit slicing slip (+1-515... vs +1-151...). Family: value extraction slip. Mostly noise.
3. task_..._b4505336 — narrow short_description search empty → gave up; target only in description. Family: stopped early after empty narrow search. Systematic.
4. task_..._63dff8c3 — (a) added unrequested description; (b) after reopening incident, didn't clear inactive assignee per policy invariant. Systematic. Golden fails (stale).
5. task_..._f89140aa — 228-char problem_statement not truncated to 225 cap. Family: skipped length rule. Golden fails (miscounted).
6. task_..._7e9733a5 — prose mapping "high elevated criticality" → wrote critical instead of high. Golden fails (omitted priority).

Ranked: missed explicit policy transform/invariant 3; over-cautious refusal 1; stopped early 1; extra field 1; extraction slip 1.
Golden reliability: goldens fail own verifiers in 4,5,6 (and likely 1); verifiers grounded in policy → stale goldens, not unfair verifiers.

Clauses: cite-a-rule-to-refuse; rule-sweep before each write (counts/mappings shown); post-change invariant check; minimal fields; broaden search before concluding absence.

## Batch C (7 tasks)

1. task_..._18cc7ee8 — KB matching set ("refers to X or Y") searched with narrow keywords, missed one article; also omitted policy-format worknotes (rule conditional → weak defect).
2. task_..._c962e7e9 — added unrequested `subject` on notification; policy says pass only named optional args.
3. task_..._5febbc10 — change → assess should cascade linked incidents to on_hold + SLA pause (buried rules); agent stopped after the change update.
4. task_..._cf718c78 — buried companion status default when updating a problem; left status unchanged.
5. task_..._26ab51da — user "wrap up" overrode policy precondition (keep in implement without verified outcome). Intentional user-vs-policy test.
6. task_..._b09b6872 — 242-char description not truncated to 240 cap.
7. task_..._ccb66a9a — user overrode policy-fixed initial status and preserved-field rule on looked-up ticket.

Ranked: missed side effect/default/companion field 3; user overrode policy 2; extra optional field 1; incomplete matching 1; length limit 1.
Clauses: policy precedence; full-policy rule sweep incl. mid-sentence clauses; cascade to related records; minimal arguments; exhaustive matching + exact formats.
