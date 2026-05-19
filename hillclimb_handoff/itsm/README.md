# ITSM Handoff

ITSM prompt-only work improved partial completion but did not improve exact
success on the full public split. A separate checklist/reconciliation workflow
later produced a better exact result.

## Current Read

| Run | Exact success | Avg verifier pass | Raw verifier pass |
| --- | ---: | ---: | ---: |
| Medium baseline | 31/103 | 44.54% | 156/375 |
| High baseline | 23/103 | 39.38% | 136/375 |
| Best prompt-only addendum | 31/103 | 49.78% | 182/375 |
| Checklist/reconciliation workflow | 39/103 | not summarized here | not summarized here |

The prompt-only addendum found useful partial-completion signal, but it traded
exact wins and losses rather than lifting the headline metric.

## Failure Modes That Mattered

- Main incident action completed while linked notifications, knowledge links,
  SLAs, CI changes, or relationship records were missed.
- Notification follow-through omitted intended recipients or exact requested
  notification semantics.
- Role, article, requester, or context ambiguity caused early stopping.
- Broad linked-state prompts improved verifier coverage but did not reliably
  convert partial repairs into exact wins.

## Prompt Evolution

The strongest prompt-only result was notification-follow-through oriented. It
improved raw verifier coverage but tied exact success. Later attempts at broad
closeout and field-fidelity prompts did not transfer on residual held-out
slices.

The stronger result came from orchestration:

1. Generate a concise requested-state checklist from the user request.
2. Run the normal agent with that checklist visible.
3. Before final answer, reconcile the checklist against tool-result evidence.
4. Continue for up to three repair iterations if explicitly requested end-state
   items are still incomplete and feasible.

## Start Here

For a new agent:

1. Do not assume another global ITSM prompt will help exact success.
2. Reproduce the prompt-only baseline and best prompt on a small slice.
3. Compare that against checklist/reconciliation on the same slice.
4. Inspect whether misses are linked-record completion, notification routing,
   knowledge/article relationships, or lifecycle-state issues.

## Next Useful Work

- Turn checklist/reconciliation into a clean orchestrator experiment that is
  easy to rerun.
- Track exact wins and exact regressions, not only raw verifier deltas.
- For prompt-only work, target one visible request pattern at a time and require
  held-out support before scaling.
