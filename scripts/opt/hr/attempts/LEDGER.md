# HR prompt-opt ledger (gpt_5p6_terra_medium, react, HR_184_Final, split 56 train / 128 test)

| attempt | overlay | step? | train | test | all | notes |
|---|---|---|---|---|---|---|
| baseline | none | no | 45/56 80.4% | 94/128 73.4% | 139/184 75.5% | 11 train failures analysed (attempts/baseline/analysis.md) |
| baseline_r2 | none | no (repeat) | 43/56 76.8% | 99/128 77.3% | 142/184 77.2% | Two-run baseline mean: train 78.6 / test 75.4 / all 76.4. Test noise is about ±4pp. |

## Leakage audits
- v1: PASS with one soft flag. Rule 4's example list matched the catalog record types in a 2-config prohibition family, so it was replaced with "supporting, parent/child, or linked configuration records". Everything else is generic (terms appear in 42–175 configs).
