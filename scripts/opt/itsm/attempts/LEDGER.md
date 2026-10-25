# ITSM prompt-opt ledger (gpt_5p6_terra_medium, react, ITSM_181_Final, split 53 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | – | 34/53 64.2% | 82/128 64.1% | 116/181 64.1% | 19 train failures diagnosed (attempts/baseline/analysis.md) |
| baseline_r2 | none | – (noise) | 36/53 67.9% | 74/128 57.8% | 110/181 60.8% | baseline 2-run mean: train 66.0 / test 60.9 / all 62.4; test run-to-run swing ~6 pts |
| step01_v1 | v1 (9-point working method, system suffix) | 1 | 38/53 71.7% | 85/128 66.4% | 123/181 68.0% | train gained 3 / lost 4 vs baseline union; 11 train tasks fail 3/3 |
| step01_v1_r2 | v1 | – (repeat) | 37/53 69.8% | 84/128 65.6% | 121/181 66.9% | v1 2-run mean: train 70.8 / test 66.0 / all 67.4 (test +5.1 vs baseline) |
| step02_v2 | v2 (visible PLAN / WRITE CHECK / FINAL AUDIT + short principles) | 2 | 39/53 73.6% | 89/128 69.5% | 128/181 70.7% | train vs v1 union: gained 6 (incl. 4 that failed all prior runs), lost 6 |
| step02_v2_r2 | v2 | – (repeat) | 40/53 75.5% | 86/128 67.2% | 126/181 69.6% | v2 2-run mean: train 74.5 / test 68.4 / all 70.2 (test +2.4 vs v1, +7.5 vs baseline) |
| step03_v3 | v3 (quoting sweep, per-param create sweep, defaults on create/update/messages, derived/related values, transform-don't-refuse, word-level counting, Follow-up after writes, probing audit) | 3 | 42/53 79.2% | 90/128 70.3% | 132/181 72.9% | train vs v2 union: gained 5, lost 7 (4 of them passed all 6 prior runs → suspect v3-caused) |
| step03_v3_r2 | v3 | – (repeat) | 41/53 77.4% | 91/128 71.1% | 132/181 72.9% | v3 2-run mean: train 78.3 / test 70.7 / all 72.9 (test +2.3 vs v2, +9.8 vs baseline) |
| step04_v4 | v4 (v3 + all-branch quoting, informal nouns, End state + ordering, two-record refs, search grid, same-meaning enum, unset-needs-search, unconditional counting, audit fixes) | 4 | 39/53 73.6% | 91/128 71.1% | 130/181 71.8% | train vs v3 union: gained 3, lost 9; first-ever pass of 48850a98 (enum mapping) |
| step04_v4_r2 | v4 | – (repeat) | 40/53 75.5% | 94/128 73.4% | 134/181 74.0% | v4 2-run mean: train 74.5 / test 72.3 / all 72.9 (test +1.6 vs v3, train −3.8: train/test divergence, within noise) |
| step05_v5 | v5 (v3 + net-positive v4 additions: all-branch quoting w/ continuations, End state + ordering, two-record refs, search grid, unconditional count, audit vs End state) | 5 | 43/53 81.1% | 87/128 68.0% | 130/181 71.8% | best train, weaker test → divergence; repeat |
| step05_v5_r2 | v5 | – (repeat) | 41/53 77.4% | 93/128 72.7% | 134/181 74.0% | v5 2-run mean: train 79.2 / test 70.3 / all 72.9. v3, v4, v5 all = 72.9 overall → content plateau; next: structure |
| step06_v6 | v6 (v5 text, policy_to_user: overlay = system msg, policy + request in user turn) | 6 | 40/53 75.5% | 91/128 71.1% | 131/181 72.4% | same band as v3–v5; no structural gain → not repeated |

## Leakage audits
- v1: audit flagged the length-limit clause (close paraphrase of a policy sentence in 4–5 tasks, encoded the verifier outcome) and, weakly, "never rely on the tool's own default" (2 tasks). Both rewritten generically; field-type list also generalized. No entity names/IDs/values.
- v7: PASS (segment-counting and level-mapping phrases occur in 0 tasks; level-mapping clause shares no distinctive words with the one priority task that motivated it — generic, but flagged as an overfitting risk to watch on test). Clarified "wording" → "policy's wording for the resulting value".
- v6: v5 text verbatim (policy_to_user structure) + header pointing at the <operating_policy>/<user_request> tags; new terms occur in 0 configs (direct grep; no subagent needed for a structural-only delta). PASS.
- v5: PASS (composed of audited v3/v4 text; only new phrase "(with any sentences that continue it)" → 0 tasks).
- v4: PASS (all 22 new phrases 0 tasks; one 4-gram shared with 1 task, unrelated meaning). Softened "set the reference field on each side" (close to a rare both-directions idea in 3 train tasks) → "check both records' fields for a reference to the other and apply any policy sentence about it".
- v3: PASS (new phrases 0–4 tasks each, none paraphrase rare policy sentences; 4-gram scan clean). Cosmetic: "Triggered:" label → "Follow-up:" (one policy uses "triggered").
- v2: audit flagged 4 near-paraphrases (≤4 tasks each): length-count convention ("spaces included"), "carries a value" inheritance wording, quoted "refers to", "text templates"; minor "does not cancel". All rewritten to generic forms ("count as the policy defines", "if the policy derives a value from another record, read it", "matching set by a criterion", "values the policy gives verbatim", "does not override"). No entity names/IDs/values.
