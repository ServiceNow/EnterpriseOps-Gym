# baseline: train failure analysis (11/56 failed)

Two subagents covered disjoint halves of the dumps in `results/opt/hr/baseline/failure_dumps/`.

## Per task
| task | root cause | family | golden passes failing verifier? |
|---|---|---|---|
| e98b68ff | Policy forbids a class of catalog records. The agent read the ban narrowly and created the categories, topics, template, criteria and service. | prohibition scope read too narrowly; user pressure | no |
| 50888c15 | The agent omitted a policy-writable field from the update and claimed the tool couldn't write it | declared field unsupported without trying | yes |
| 0251799d | On case reassignment it never set the case's pending notification to ready | policy-triggered side effect missed | no |
| b51123d7 | Policy fixes a field value. The agent wrote the value the user stated instead. | user over policy (fixed value) | no |
| bce7688f | Assigning requires a notification to the assignee (type and status set by the rule). It wasn't sent. | policy-triggered side effect missed | no |
| 7e1ab92b | Same as 0251799d: reassignment means release the pending notification | policy-triggered side effect missed | no |
| eba3a1c5 | Same reassignment notification release missed | policy-triggered side effect missed | no |
| 4bdb9321 | Stopped after create. Never completed the task, closed the case or notified, and the assignee alert was missing. | stopped early, chained steps treated as deferred; side effect missed | no (alert verifier) |
| 196fe24f | Policy fixes the active flag on create. The agent used the tool default. | policy fixed value vs tool default | no |
| e1d3c37d | Ambiguous name. Asked the user instead of narrowing with group-member lookups. 0 writes. | stopped early / asked user | no (worknote verifiers) |
| 7b06ae45 | Policy default COE overridden by a value inferred from the service. Mandatory companion task missing. | policy default vs inferred value; companion record missed | no |

## Families (ranked)
1. Side effects or companion records the policy triggers were missed: 6 tasks
2. A policy-fixed value lost out to the user, a tool default, or an inferred value: 3 tasks
3. Stopped early, deferred steps, or asked the user when lookups could resolve it: 2 tasks
4. A prohibition's scope was read narrowly: 1 task
5. A field was declared unsupported without trying: 1 task

## Noise vs defects
None of these look like sampling noise. Verifiers match the policy text. Goldens fail in 10/11 tasks, so they look stale (policy rules were probably added after the goldens were recorded). Treat goldens as hints, not ground truth.

Phrase frequency across all 184 system prompts: notifications mentioned in 116; "on notice" in 8; release/ready in 4; "by default" in 8.
