# HR Handoff

HR benefited from a shared prompt, then improved further with visible
request-shape routing.

## Current Read

| Run | Exact success | Avg verifier pass | Runtime |
| --- | ---: | ---: | ---: |
| Medium baseline | 16/102 | 42.34% | 38.37s/task |
| High baseline | 17/102 | 49.68% | 131.56s/task |
| Best shared prompt | 30/102 | 67.19% | 43.44s/task |
| Visible-text routed prompt | 36/102 | 71.56% | 44.23s/task |

High was one task ahead at baseline but far slower. Medium/default became the
better optimization track.

## Failure Modes That Mattered

- Policy-literal premature stopping under long HR prompts.
- Case created or updated without the exact requested final state.
- Dropped persisted fields such as account/reference number, source, priority,
  assignment group, assignee, service, or state.
- Missing linked records such as knowledge, notifications, approvals, routing
  rules, skills, or fulfillment follow-through.
- Prompt changes that helped case-field workflows but regressed broader HR
  workflows when applied everywhere.

## Prompt Evolution

The shared prompt focused on:

- resolving person, service, and group records,
- one broader verified lookup after exact miss,
- preserving urgency, priority, assignment, service, and state,
- completing knowledge and notification follow-through,
- checking every explicitly requested record type before finishing.

The later improvement came from visible-text routing: use a case-field fidelity
addendum only when the user request visibly asks for HR case create/update work
and includes at least two persisted case-field cues. Otherwise use the default
shared addendum.

## Start Here

For a new agent:

1. Start from the shared HR addendum.
2. Apply the case-field addendum only through visible request-shape routing.
3. Do not route using hidden task IDs, prompt-family IDs, verifier details, or
   answer leakage.
4. Inspect remaining misses by workflow shape: case fields, setup graph,
   knowledge/document linkage, notifications, approvals, and profile/access
   changes.

## Next Useful Work

- Tighten visible routing only if a new request-shape cluster is supported by
  held-out evidence.
- Avoid stacking many narrow fixes into the global HR prompt.
- Consider a lightweight checklist artifact if exact final-state checks keep
  failing after prompt routing.
