# Hybrid Handoff

Hybrid is less mature than Teams, Calendar, CSM, HR, and ITSM. Treat it as an
early-stage continuation point.

## Current Read

| Run | Exact success | Avg verifier pass | Raw verifier pass |
| --- | ---: | ---: | ---: |
| Medium baseline, full public split | 24/88 | 55.20% | 221/371 |
| Variant A, dev12 | 2/12 | 48.26% | 31/62 |
| Variant B, dev12 | 2/12 | 48.41% | 32/62 |
| Variant A, held-out8 | 2/8 | 61.77% | 28/40 |
| Variant B, held-out8 | 2/8 | 60.21% | 27/40 |

The first prompt loop improved partial completion but did not create new exact
wins on the small sets. Do not scale the first variants to the full public split
without more evidence.

## Failure Modes That Mattered

- One side of the two-domain task completed while the other side missed exact
  persisted state.
- A refusal or lookup failure in one domain erased downstream work in the other.
- Multi-record and multi-system bundles were under-completed.
- Some failures were inherited from constituent domains such as Calendar exact
  event state, Email filter shape, CSM lookup behavior, or ITSM linked-record
  follow-through.
- A few tasks showed likely harness or verifier issues and should be separated
  from prompt work.

## Prompt Evolution

Two compact first-loop variants were tested:

- Variant A: cross-domain completion, preserving values resolved in one system
  and finishing all requested downstream effects.
- Variant B: bounded recovery, avoiding collapse of the whole request after one
  narrow lookup miss or blocked subtask.

Variant A is the cleaner current seed because it matches the dominant failure
shape and transferred slightly better on held-out.

## Start Here

For a new agent:

1. Start with the Hybrid dataset overview and baseline failure shape.
2. Use Variant A as the current manual seed, not a final prompt.
3. Build residual-balanced dev and held-out slices before another edit.
4. Separate inherited single-domain failures from true cross-domain handoff
   failures.
5. Do not write a long universal Hybrid controller prompt unless residuals prove
   the issue is general handoff discipline.

## Next Useful Work

- Run GEPA seeded from Variant A on a residual-balanced split.
- Or make one tiny manual edit if a single near-complete residual pattern is
  clearly supported.
- Keep the analysis explicit about whether each failure is Calendar, Email, CSM,
  HR, ITSM, Teams, or genuinely Hybrid.
