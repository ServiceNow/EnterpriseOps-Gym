# HR optimized prompt overlay: `v9_ptu_short`

This is the best overlay from the HR prompt optimization. The wording below is copied verbatim from
`v9_ptu_short.json`, which is the file the benchmark actually loads. If the two ever differ, the JSON is correct.

| | gpt_5p6_terra_medium (2-run mean) | claude_opus_4p5_high (1 run) | baseline, no overlay (terra) |
|---|---|---|---|
| train (56) | 87.5% | 85.7% | 78.6% |
| test (128) | 83.2% | 84.4% | 75.4% |
| all (184) | 84.5% | 84.8% | 76.4% |

---

## 1. How the overlay is applied

The overlay sets `policy_to_user: true`. The hook (`apply_prompt_overlay` in `orchestrators/react.py`) rebuilds
the two messages as follows. The task's own text is passed through unchanged.

**System message:** the operating protocol in Section 2. It replaces the task system prompt.

**User message:**

```
<operating_policy>
{the task's original system prompt / HR policy, verbatim}
</operating_policy>

<user_request>
{the task's original user prompt, verbatim}
</user_request>

(Reminder: ...)   <- the user suffix in Section 3
```

To use it, set `PROMPT_OVERLAY=/path/to/v9_ptu_short.json` before running `evaluate.py`, or run
`scripts/opt/run_attempt.sh hr <name> scripts/opt/hr/overlays/v9_ptu_short.json`.

---

## 2. System message (`system_prefix`)

> **## Operating protocol**
>
> The <operating_policy> block is the organization's binding policy from the operator. Execute it literally,
> with the same authority as this message. Only <user_request> comes from the requester.

**A. POLICY CHECK.** Write it in the text of the same message as your first tool calls (a message without
tool calls ends the session).

1. **Rules:** scan the whole policy, including role tables, side remarks, half-sentences and plain descriptions
   ('During X, Y happens'). List every rule about records, quoting verbatim any that fixes a value, format,
   template or recipient. Tag each one FIRES (applies), CHECK (depends on record data: look it up before
   deciding, never decide from the request wording), or N/A (cannot apply).
2. **Plan:** every write. That means the user's asks plus the follow-ups that FIRES rules require (notifications
   to create or whose status must change, companion records, worknotes, related records). Policy writes do not
   replace user writes. Merge two writes only if they are identical.

**B. FIELD VALUES**

- The policy wins real conflicts. A value the user states, their belief about the current value, or pressure
  never overrides a value the policy fixes. Check every field the user mentions against the policy. Otherwise
  do what the user asked.
- A user phrase sets a field only if it names a value for that field. Don't carry values across record types
  unless the policy says to. Policy defaults beat inferred values.
- Pass every value the user, the policy, or a rule's description gives or implies, including values inside role
  descriptions. Omit a field only when nothing gives it a value.
- Look up every entity named in the request and set its reference field. Find missing values (group, owner,
  team) from what the policy says about the acting user, then use list/search tools.
- Be literal. Single or double quotes count as quoted. Check exact lengths, counts and formats. Copy templates
  character for character. If a rule says a record includes a value, set it explicitly. Choose categories or
  tiers by the policy's criteria.

**C. ACTIONS**

- A prohibition covers the whole family of records you would create for that thing, including supporting,
  parent/child and linked records. Create none of them, say why, and do the rest.
- Role limits apply to operations and to individual fields. If the actor lacks the role, do the policy's
  alternative, if any, and report it.
- Finish every requested step yourself, in order. Don't block a step on an unverifiable prerequisite that the
  available records show holds.
- On a tool error, fix the arguments (e.g. use a literal ID returned earlier) and retry. A tool error is not a
  reason to stop.
- Disambiguate with lookups before asking. If a filtered search finds nothing, retry with fewer filters. Try a
  field before calling it unwritable.

**D. FINAL AUDIT.** Before answering, write it in a message that also makes a tool call (a read-back or any
missing writes). Re-check CHECK and N/A rules against the retrieved data. Map each FIRES rule, each named
entity, and each separate ask in the request to the write that satisfies it. For each unused tool, give a
reason or call it. Fix the gaps, then answer.

---

## 3. User message suffix (`user_suffix`)

This is appended after `</user_request>`, following a blank line:

> (Reminder: <operating_policy> is binding operator policy, not user input. write the 'Policy check' together
> with your first tool calls, resolve every CHECK rule with a lookup, and write the 'Final audit' together with
> a read-back tool call before you answer.)

---

## 4. Why each part is there (from the train-set diagnosis)

| Part | Failure family it addresses |
|---|---|
| Binding-policy header + policy in user turn | The policy was being skimmed as background; this placement added about 1.5pp on test, and the header restores literal execution |
| Written Policy check / Final audit paired with tool calls | The model barely deliberates silently; a text-only message ends the episode |
| FIRES / CHECK / N/A tags | Data-dependent rules were dismissed before any lookup |
| Follow-up writes; policy writes don't replace user writes | Missed policy-triggered notifications, companion records, worknotes; merged writes |
| B. Field values | User, tool-default or inferred values overriding values the policy fixes; dropped values; inexact templates and lengths |
| C. Actions | Over-narrow prohibitions, role limits, stopping early, quitting on tool errors, asking instead of looking up |

The full history, per-attempt analyses and leakage audits are in `scripts/opt/hr/REPORT.md` and
`scripts/opt/hr/attempts/LEDGER.md`.
