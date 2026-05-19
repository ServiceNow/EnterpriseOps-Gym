# Hillclimb Handoff Notes

This folder is a compact handoff map for continuing EnterpriseOps-Gym hillclimb
work. It is intended for a new agent or teammate who is starting from this
GitHub repo and needs to understand where prior experiments found signal.

It does not include raw result JSONs or duplicate long-form analysis logs. Each
dataset subfolder gives the current read, the failure modes to keep in mind, and
the next useful action.

## How To Start

1. Read this file.
2. Pick the dataset you are continuing.
3. Read that dataset's `README.md` under this folder.
4. Reproduce a small slice before changing prompts or orchestrators.
5. Keep changes scoped: one hypothesis, one eval slice, one result note.

## Dataset Map

| Dataset | Start here | State |
| --- | --- | --- |
| Teams | [`teams/README.md`](teams/README.md) | Clearest successful hillclimb. Use as the reference pattern. |
| Calendar | [`calendar/README.md`](calendar/README.md) | Prompt improvements found real signal, but residuals are family-specific. |
| CSM | [`csm/README.md`](csm/README.md) | Small exact prompt gain; high-reasoning failures are a separate tool-call behavior signal. |
| HR | [`hr/README.md`](hr/README.md) | Shared prompt helped; visible request-shape routing helped further. |
| ITSM | [`itsm/README.md`](itsm/README.md) | Prompt-only gains improved verifier coverage, not exact success. Checklist orchestration did better. |
| Hybrid | [`hybrid/README.md`](hybrid/README.md) | Early-stage. Main issue is often constituent-domain exact state after cross-domain handoff. |

## Cross-Dataset Lessons

The portable lessons are:

- Verify entities, but do one broader lookup before giving up on an obvious
  shortened or partial business reference.
- Decompose the user request into atomic end-state requirements before acting.
- Finish independent subtasks even if one subtask is blocked.
- Optimize for final persisted state, not just a plausible tool call.
- Preserve exact user-stated fields, names, relationships, recipients, and
  lifecycle states.
- Avoid broad "do more" prompt text once the residual errors are domain-specific.

Teams is the best reference case because it shows the full loop:

1. baseline,
2. failure analysis,
3. targeted prompt variants,
4. held-out validation,
5. full public-split run,
6. post-run failure audit.

For the other datasets, the same information exists as summarized notes here,
but not always as a single polished package.

## Recommended Working Loop

Use this loop for any new work:

1. State the hypothesis in one sentence.
2. Select a small dev slice and a held-out/regression slice.
3. Run the current baseline or current best prompt.
4. Inspect failed traces before editing prompts.
5. Write the smallest prompt or orchestration change that targets the observed
   failure.
6. Promote only if the held-out check supports it.
7. Scale to the public split only after the small-set signal is credible.
8. Write a short result note that records the exact run, metric deltas, and
   failure modes.

## Caveats

- These notes summarize local experiments. They are not official leaderboard
  claims.
- Most runs used the released public split, not non-public held-out data.
- Some comparisons changed more than one variable, such as reasoning effort and
  API surface. Treat those as operational findings, not pure ablations.
- Do not assume a Teams prompt transfers wholesale to another dataset. Reuse the
  general behavior, then re-derive the domain-specific details from traces.
