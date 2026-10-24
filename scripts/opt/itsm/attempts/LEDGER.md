# ITSM prompt-opt ledger (gpt_5p6_terra_medium, react, ITSM_181_Final, split 53 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | – | 34/53 64.2% | 82/128 64.1% | 116/181 64.1% | 19 train failures diagnosed (attempts/baseline/analysis.md) |

## Leakage audits
- v1: audit flagged the length-limit clause (close paraphrase of a policy sentence in 4–5 tasks, encoded the verifier outcome) and, weakly, "never rely on the tool's own default" (2 tasks). Both rewritten generically; field-type list also generalized. No entity names/IDs/values.
