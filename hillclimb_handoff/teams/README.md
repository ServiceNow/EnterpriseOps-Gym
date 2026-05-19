# Teams Handoff

Teams is the cleanest successful hillclimb and should be the reference case for
new work.

## Current Read

The strongest Teams result came from the target model on the default or
medium-effort path plus a compact Teams-specific prompt addendum.

| Run | Scope | Exact success | Avg verifier pass | Runtime |
| --- | --- | ---: | ---: | ---: |
| Published reference model | Teams domain | 31.0% | 66.3% | not summarized |
| Local target baseline, default/medium | 61 public tasks | 24/61 | 65.46% | 15.9s/task |
| Local target baseline, high reasoning | 61 public tasks | 21/61 | 56.22% | 119.5s/task |
| Target + final Teams addendum, default/medium | 61 public tasks | 36/61 | 85.45% | 17.2s/task |

The working conclusion was that default/medium execution was better for this
benchmark than high reasoning. High reasoning tended to over-elaborate low-level
payloads and spend time exploring simulator API details that the verifier did
not reward.

## Failure Modes That Mattered

- Exact lookup failure caused early aborts instead of one broader recovery
  lookup.
- Calls were created with the wrong persisted source or target shape.
- Virtual-event creation sometimes missed exact creator, co-organizer, status,
  audience, or time fields.
- Create retries sometimes produced duplicate rows after a tool error.
- Some channel, tab, message, and membership edits were semantically right but
  verifier-wrong in exact naming, role, or content.

## Prompt Evolution

The successful addendum combined:

- one broader lookup after exact lookup miss,
- atomic requirement tracking,
- completion of independent subtasks,
- direct-call source and target discipline,
- exact virtual-event field handling,
- deterministic construction of required non-identity config values.

Manual variants were tested first. GEPA was then used as a validation and
refinement loop, but did not beat the best manual seed.

## Start Here

For a new agent:

1. Reproduce a small Teams slice with the current baseline.
2. Compare against the final Teams addendum.
3. Inspect failures before changing the prompt.
4. If continuing Teams, focus on call shape, virtual-event idempotency, and
   exact text/name/role mismatches.
5. If transferring lessons to another dataset, copy only the general execution
   behaviors. Re-derive schema-specific rules from that dataset's traces.

## Next Useful Work

- Add a typed tool or orchestration layer for calls so caller, recipients,
  callback configuration, and idempotency are handled outside prompt text.
- Label the remaining failures by whether they are promptable, tool-shape
  issues, verifier issues, or run-to-run variance.
- Use Teams as the reference package format for the other datasets if a fuller
  handoff package is needed.
