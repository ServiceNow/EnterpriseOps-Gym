# HR prompt-opt ledger (gpt_5p6_terra_medium, react, HR_184_Final, split 56 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | no | 45/56 80.4% | 94/128 73.4% | 139/184 75.5% | 11 train failures analysed (attempts/baseline/analysis.md) |
| baseline_r2 | none | no (repeat) | 43/56 76.8% | 99/128 77.3% | 142/184 77.2% | Two-run baseline mean: train 78.6 / test 75.4 / all 76.4. Test noise is about ±4pp. |
| step01_v1 | v1 | 1 | 46/56 82.1% | 102/128 79.7% | 148/184 80.4% | 9-rule protocol appended to the system prompt. Train gained e1d3c37d; lost 4 tasks vs the baseline union. The 6 targeted tasks still fail 3/3. |
| step01_v1_r2 | v1 | no (repeat) | 46/56 82.1% | 104/128 81.2% | 150/184 81.5% | v1 two-run mean: train 82.1 / test 80.5 / all 81.0, +5.1pp test over baseline |
| step02_v2 | v2 | 2 | 46/56 82.1% | 102/128 79.7% | 148/184 80.4% | Visible Policy check and Final audit plus regression fixes. Train churn vs v1: +5 (727d852c, 9e1a6b99, bce7688f, 07f8ca42, 7e1ab92b) / −8. |
| step02_v2_r2 | v2 | no (repeat) | 45/56 80.4% | 104/128 81.2% | 149/184 81.0% | v2 two-run mean: train 81.2 / test 80.5 / all 80.7. Ties v1; the gains were offset by 5 overlay-caused losses (attempts/step02_v2/analysis.md). |
| step03_v3 | v3 | 3 | 48/56 85.7% | 105/128 82.0% | 153/184 83.2% | FIRES/CHECK/N/A tags, full rule scan, positive field rule, scoped policy-over-user. Best single run so far. |
| step03_v3_r2 | v3 | no (repeat) | 48/56 85.7% | 103/128 80.5% | 151/184 82.1% | v3 two-run mean: train 85.7 / test 81.2 / all 82.6. Current best. |
| step04_v4 | v4 | 4 | 50/56 89.3% | 103/128 80.5% | 153/184 83.2% | Request ledger with cited calls, unused-tool check, per-sentence event tagging, per-field cross-check. Train up but test flat (possible train/test divergence). |
| step04_v4_r2 | v4 | no (repeat) | 46/56 82.1% | 104/128 81.2% | 150/184 81.5% | v4 two-run mean: train 85.7 / test 80.9 / all 82.3. Ties v3; content plateau around 81% test. v3 kept as base (shorter, slightly higher test). |
| step05_v5_prefix | v5_prefix | 5 | 49/56 87.5% | 105/128 82.0% | 154/184 83.7% | v3 text placed before the policy (system_prefix), user_suffix reminder kept |
| step06_v6_ptu | v6_ptu | 6 | 47/56 83.9% | 108/128 84.4% | 155/184 84.2% | v3 text as the system message; task policy moved into the user turn (`<operating_policy>`). Best single test run. |

## Leakage audits
- v1: PASS with one soft flag. Rule 4's example list matched the catalog record types in a 2-config prohibition family, so it was replaced with "supporting, parent/child, or linked configuration records". Everything else is generic (terms appear in 42–175 configs).
- v2: FLAGGED, then fixed before the run.
  - HIGH: a "descriptive wording is not a value / case status is not notification status" example. It quoted a user phrase found in 6 configs and contradicted 1 test policy; replaced with a generic cross-record-type rule.
  - MEDIUM: "not done through this assistant" paraphrased a policy idiom in 3–5 configs; replaced with "forbids creating".
  - LOW–MEDIUM: "raising a request" echoed one train prompt; removed.
  - LOW: "which team you belong to" and "narrow it down with the other details" neutralized. The JSON was re-validated and none of the flagged phrases remain.
- v5_prefix / v6_ptu: v3 text verbatim, placement only. The only change is the phrase "policy above", now "policy below" or "operating policy given in the user message". The v3 audit carries over.
- v4: PASS. Optional rewrite applied: the synonym examples ("transfer / change of owner / registering / opening"), which may trace back to one train task, were replaced with a neutral "different verb for the same event" wording.
- v3: PASS. One optional rewrite applied: "classification levels" was closest to a single train task, so it now reads "categories or tiers by criteria". New phrases are generic (0–57 config hits, no entity or value content). Rejected subagent wording that mirrored a case status into a notification status (flagged in the v2 audit).
