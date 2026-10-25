# HR prompt-opt ledger (gpt_5p6_terra_medium, react, HR_184_Final, split 56 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | no | 45/56 80.4% | 94/128 73.4% | 139/184 75.5% | 11 train failures analysed (attempts/baseline/analysis.md) |

## Leakage audits
- v1: PASS with one soft flag. Rule 4's example list matched the catalog record types in a 2-config prohibition family, so it was replaced with "supporting, parent/child, or linked configuration records". Everything else is generic (terms appear in 42–175 configs).
