# CSM Handoff

CSM produced a small exact-success prompt gain and a clearer verifier-coverage
gain on medium/default execution. It also produced the clearest high-reasoning
tool-call failure signal.

## Current Read

| Run | Exact success | Avg verifier pass | Runtime |
| --- | ---: | ---: | ---: |
| Medium baseline | 38/103 | 64.20% | 47.35s/task |
| High baseline | 2/103 | 15.13% | 238.24s/task |
| Best promoted prompt-only run | 39/103 | 65.79% | 44.66s/task |

Medium/default was the better track. High reasoning often turned simple record
lookups into over-constrained searches, then stopped after its own guessed
filters hid the target record.

## Failure Modes That Mattered

- Over-constrained lookup filters on known case numbers, serials, products,
  accounts, or locations.
- Premature refusal after a recoverable lookup miss.
- Completing one case or record well while leaving the same requested bundle
  incomplete for other targets.
- Missing linked state after the main case update, such as group membership,
  reassignment, knowledge links, resolution associations, or notifications.
- Substituting nearby notification verbs instead of preserving the user's
  requested communication action.

## Prompt Evolution

The prompt work converged on bounded recovery and exact bundle completion:

- keep direct record requests direct,
- do one broader verified lookup after an exact miss,
- preserve exact lifecycle state, assignment, knowledge, and communication
  wording,
- complete repeated bundles for every named target,
- finish independent linked-record work after the main case action succeeds.

High-reasoning rescue prompts did not fix the dominant issue. Even with explicit
sparse-lookup examples, high reasoning continued to fill optional filters with
guesses. That points to tool-call construction behavior, not just prompt wording.

## Start Here

For a new agent:

1. Treat medium/default as the main optimization path.
2. Preserve the bounded-recovery and bundle-completion addendum as the current
   best prompt direction.
3. Do not restart high-reasoning full runs until the tool-call argument behavior
   is understood.
4. When inspecting failures, separate lookup construction, case lifecycle,
   linked knowledge, membership, and notifications.

## Next Useful Work

- Add a sparse lookup contract or typed primary-key lookup tools for CSM records.
- Re-run a small diagnostic slice before scaling high reasoning again.
- Keep prompt changes small; the best gains came from execution discipline, not
  broad instruction stacking.
