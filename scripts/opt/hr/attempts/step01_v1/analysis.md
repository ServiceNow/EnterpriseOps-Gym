# step01_v1: analysis (train 46/56, test 102/128)

## Why the targeted tasks still fail (6 tasks, failed 3/3 runs)
- The model barely deliberates: about 15–225 reasoning tokens per turn, empty `content` on tool-call messages, and parallel writes immediately after `get_user`. Silent instructions ("read the whole policy", "plan", "re-read rule by rule") are skipped. Concrete action clauses did work (finish the sequence; assignee notification; companion record).
- Harness constraint: an AI message with no tool calls ends the episode (`orchestrators/react.py`). A checklist has to be written in the same message as tool calls.
- Per task:
  - e98b68ff: noticed the prohibition, scoped it to a single record type.
  - bce7688f: fixed the notification, but stochastically omitted a service lookup (reference field null).
  - 7e1ab92b: never noticed the reassignment notification-release rule.
  - 4bdb9321: improved from 2–3/6 to 5/6 verifiers. Copied the case status into the notification status because of a descriptive user phrase.
  - 196fe24f: never noticed an active-flag rule buried in a role paragraph.
  - 7b06ae45: noticed the companion-task rule but declined it for lack of a group; ignored the policy default COE in favour of an inferred one. Partly a benchmark defect (garbled clause).

## Regressions vs the baseline union (4 tasks)
- 07f8ca42: **overlay-caused.** The tool rejects the policy's runtime-reference format. Baselines retried with the literal ID and passed. v1 stopped after one error, citing "unless a tool error blocks them".
- 9e1a6b99: **probably overlay-caused.** A role-restricted operation was done directly instead of via the policy's alternative (a request task). Clauses 5 and 7 pushed toward doing the write.
- 727d852c: noise. A filtered knowledge search returned 0 results and the agent gave up; one baseline passed by listing without filters.
- b51123d7: noise (1/2 in baseline). A policy value sits in the second half of an unrelated conditional sentence.

## v2 changes
- A visible "Policy check" (triggers and field sources), written together with the first tool calls.
- A visible final audit, written together with a read-back tool call before the final answer.
- A tool error is not a stop reason: correct the arguments and retry.
- Role-restricted operations get the policy alternative.
- Field ownership: descriptive phrases are not values; do not copy fields across records; policy defaults beat inferred values; do not set unrequested flags.
- Missing values are resolved from policy self-descriptions plus lookups; searches are broadened before concluding a record is absent.
- A short reminder goes in user_suffix.
