# ITSM prompt-opt ledger (gpt_5p6_terra_medium, react, ITSM_181_Final, split 53 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | – | 34/53 64.2% | 82/128 64.1% | 116/181 64.1% | 19 train failures diagnosed (attempts/baseline/analysis.md) |
| baseline_r2 | none | – (noise) | 36/53 67.9% | 74/128 57.8% | 110/181 60.8% | baseline 2-run mean: train 66.0 / test 60.9 / all 62.4; test run-to-run swing ~6 pts |
| step01_v1 | v1 (9-point working method, system suffix) | 1 | 38/53 71.7% | 85/128 66.4% | 123/181 68.0% | train gained 3 / lost 4 vs baseline union; 11 train tasks fail 3/3 |

## Leakage audits
- v1: audit flagged the length-limit clause (close paraphrase of a policy sentence in 4–5 tasks, encoded the verifier outcome) and, weakly, "never rely on the tool's own default" (2 tasks). Both rewritten generically; field-type list also generalized. No entity names/IDs/values.
- v2: audit flagged 4 near-paraphrases (≤4 tasks each): length-count convention ("spaces included"), "carries a value" inheritance wording, quoted "refers to", "text templates"; minor "does not cancel". All rewritten to generic forms ("count as the policy defines", "if the policy derives a value from another record, read it", "matching set by a criterion", "values the policy gives verbatim", "does not override"). No entity names/IDs/values.
