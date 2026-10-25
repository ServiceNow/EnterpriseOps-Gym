# step02_v2: analysis (train 46/56, test 102/128)

v2 recovered 5 train tasks vs v1: 727d852c (search broadening), 9e1a6b99 (role limits), bce7688f, 07f8ca42 (tool-error retry) and 7e1ab92b (reassignment notification). It lost 8.

No early terminations and no step limits. Every run wrote the Final audit with a read-back call. The largest run made 39 calls (v1: 31–33).

## Losses
| task | cause | clause |
|---|---|---|
| 3b89aab7 | A data-dependent rule condition (assignee's group) was tagged N/A from request wording; never looked up | A.1 up-front tagging |
| 0251799d | A descriptive rule with no trigger keyword was never listed; audit said notification tools "not needed" | A.1 keyword filter |
| 11289326 | Notification sent without the status the rule describes (null stored) | B.4 "don't pass optional flags" |
| 920d6fbe | Refused a requested closure because a prerequisite couldn't be verified (new case, no tasks); notification status dropped | B.1 overreach + B.4 |
| e1d3c37d | A profile-attribute condition was tagged "N/A pending lookup"; lookup never done | A.1 up-front tagging |
| c7e85ec1 | Classification matrix misread toward the anti-user value | mostly noise, B.1 nudge |
| 559da326 | Off-by-one truncation length | noise |
| 196fe24f | Single-quoted name not treated as quoted; B.4 also stripped active flags elsewhere | noise + B.4 |

## v3 changes
- Trigger scan covers every statement of what happens to records, not keyword matches. Restores v1's follow-up-write list.
- Third tag CHECK for data-dependent conditions: look up before deciding. The Final audit re-evaluates N/A/CHECK items.
- B.4 replaced: pass every value the user, the policy or a rule description implies; omit only if nothing gives a value.
- "Policy beats user" scoped to real value conflicts. Don't block a requested step on an unverifiable prerequisite when the available records show it holds. Apply classification definitions literally.
- Literal fidelity: single or double quotes both count as quoted; check exact lengths and counts.
- Policy check shortened: triggers plus plan, no per-field source table.
- Rejected subagent wording "notification status mirroring case status" (leakage-flagged in the v2 audit, conflicts with one test policy).
