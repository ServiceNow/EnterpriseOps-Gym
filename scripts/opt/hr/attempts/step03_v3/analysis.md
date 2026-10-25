# step03_v3: analysis (train 48/56, test 105/128)

Pass counts are over 7 runs (baseline ×2, v1 ×2, v2 ×2, v3).

| task | passes/7 | cause | class |
|---|---|---|---|
| e98b68ff | 0 | Prohibition scope: v3 withholds the service and skill but builds the category, details, template and criterion | ambiguous scope / benchmark-adjacent; leave alone |
| 7b06ae45 | 0 | Needed group comes from a garbled policy fragment; never looked up | mostly benchmark defect; leave alone |
| 196fe24f | 1 | Rule buried in a role paragraph; set active=true | attention noise |
| 7e1ab92b | 5 | Explicit "give it to X" request merged into another step; audit falsely claimed done; tool never called | fixable (request ledger, unused tools) |
| 0251799d | 4 | Visible checklist dismissed the notification rule as N/A by topic (user didn't mention notifications). The silent v1 passed. | fixable (no topic-level N/A; event translation) |
| b51123d7 | 4 | Policy value in the second half of a sentence; user value used | noise; fixable (per-field policy cross-check) |
| 679067bc | 4 | Field-level role limit (only role X may set an attribute); admin set it | fixable (role limits apply to fields) |
| 627386b2 | 6 | Created the article but never linked it; link tool exposed but unused | noise (unused-tool check) |

Cross-cutting: the missing step is often an exposed tool that was never called, and the Final audit is self-reported without call citations.

## v4 changes
- Final audit becomes a request ledger: atomic verb+object actions, each mapped to the tool call or returned ID that did it. No citation means not done.
- Unused-tool check: name the clause each unused tool would serve; call it if one exists.
- Rules tagged sentence by sentence by trigger event; translate own actions into the policy's event wording; no N/A by topic; read linked records when a rule changes them.
- Per-field cross-check before writes: policy-fixed values and role limits apply to individual fields too.
- A1 prose trimmed.

Structural variants (policy_to_user, prefix) deferred. policy_to_user risks weakening policy authority on policy-beats-user tasks.
