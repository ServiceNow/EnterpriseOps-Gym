# step02_v2 train analysis

## Regressions vs v1 (6)
Blocks (PLAN / WRITE CHECK / FINAL AUDIT) were written in all 6; the audit rubber-stamped the plan every time.
- be0bd794 — buried notification-status default omitted; mostly noise (1/5 pass) but same type as cf718c78 (v1's standalone "defaults are mandatory" removed).
- 5293f595 — v2-caused: format/prefix rule treated as a block ("blocked"/"unmet precondition" wording; v1 "transform, don't refuse" removed).
- a844a76e — v2-caused: "restating the user's story is not a source" dropped a structured field (channel) the user's narrative supports. Passed 4/4 before.
- 9c6f3e01 — noise: digits spelled correctly, grouped wrongly. Suggest round-trip check of formatted value.
- 7e9733a5 — v2-caused: tool-schema fallback written as "policy default" (priority=planning) instead of deriving from impact×urgency (v1's derived-values line removed).
- cf718c78 — mostly v2-caused: buried update-time status default missed; PLAN "status of related tickets" + "preserves a field" framed status as preserve.

Proposals v3: defaults mandatory on create AND update AND messages, tool-schema defaults are not policy defaults; restore derived-values mapping; user-stated facts are a valid source for the structured field recording them; restore transform-don't-refuse; formatted-value round-trip; PLAN item for the written record's own status; FINAL AUDIT compares to policy text, not to PLAN.

## Persistent failures (8)
Block compliance: PLAN 53/53 (before first write), WRITE CHECK 53/53, FINAL AUDIT 52/53 (in a tool-calling reply); no text-only early endings; turns ~6.0 → 6.8. Audit almost never produces a corrective write (2/53). "Policy duties" filled by category label ("SLA: none") without quoting rules; unused tools waved through ("not requested").
- dc6af68f — missed (indirect wording, standalone between sections); SLA tools dismissed in audit.
- 48850a98 — missed buried create default (enum mapping); partly noise.
- 8cb006f5 / 18cc7ee8 — follow-on note stated as a fact after a link write; "Notes: none". (18cc7ee8 KB matching now correct.)
- 63dff8c3 — invented description sourced "user's report"; missed event-triggered duty on caller's other tickets (never listed).
- f89140aa — counted sentence-by-sentence: 227 vs true 228 → cut wrong. Needs word-level tally + recount.
- fad8a09d — overlay-caused: "remove any argument without a source" dropped a field the policy states as a fact; also invented problem_statement.
- ccb66a9a — found and misapplied: extra short_description; user's status value beat policy's positional value.
Proposals: quoting sweep per action/event incl. related records of same caller/CI/group; per-parameter sweep on create; free-text only dictated text or policy template; "Triggered:" re-scan after each write; audit probes unused read tools on changed records/people; word-level counting with recount.
