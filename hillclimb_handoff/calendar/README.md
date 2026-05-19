# Calendar Handoff

Calendar improved from prompt work, but the remaining misses are more
family-specific than Teams.

## Current Read

| Run | Exact success | Avg verifier pass | Runtime |
| --- | ---: | ---: | ---: |
| Medium baseline | 24/61 | 75.80% | 53.49s/task |
| High baseline | 24/61 | 74.33% | 111.0s/task |
| Best promoted prompt | 28/61 | 75.97% | 41.06s/task |

Medium was the better optimization baseline because it tied high on exact
success while running much faster.

## Failure Modes That Mattered

- Missing exact persisted Calendar state across event fields, attendees,
  recurrence, ACLs, reminders, calendar-list state, and metadata.
- Completing the headline event action but dropping a secondary surface.
- Preserving plausible schedule semantics while missing verifier-exact fields.
- Shared prompts helping some families while regressing others.

## Prompt Evolution

The useful prompt work moved from general checklist language toward
Calendar-specific persisted-state alignment.

The best result came from a compact state-alignment seed that told the model to
track the final Calendar surfaces rather than just the headline event action.
GEPA helped identify there was still signal after manual tuning, but the best
final seed was a human-compressed prompt derived from contrastive analysis.

## Start Here

For a new agent:

1. Start from the best compact state-alignment prompt.
2. Read residual failures by prompt family before writing another global prompt.
3. Keep dev and held-out slices family-aware, because Calendar prompt families
   behave differently.
4. Treat recurrence, attendees, ACLs, metadata, and calendar-list visibility as
   separate persisted surfaces.

## Next Useful Work

- Build narrower family-aware prompts or routes instead of another universal
  Calendar addendum.
- Separate exact event construction failures from calendar metadata / ACL /
  reminder failures.
- Use GEPA only after the seed and validation split are shaped by human failure
  analysis.
