# step06_v6_ptu: analysis (train 47/56, test 108/128)

v3 protocol as the system message, with the task policy moved into the user turn. Losses were compared with v3, v3_r2, v5_prefix and v5_prefix_r2.

## Likely structure-caused (passed in 3–4 of the 4 earlier runs)
- 11289326: the rule says the notice includes a value; the field was omitted (the Policy check summarized the rule).
- a1280fa7: policy template label dropped from short_description.
- 923e672e: a policy-required notice to a second recipient was merged into the user-requested one.
- 7e1ab92b: a user-requested step was dropped; the plan listed only policy-driven steps.

What they share: the policy is read as user-supplied reference material and summarized, not executed literally. The policy is not losing authority to the user; it is losing fidelity.

## Noise or hard tasks (fail at similar rates in every variant)
b51123d7, 679067bc, 7b06ae45, e9192d7e, e98b68ff.

## Hypotheses for why ptu helps on test
Recency and adjacency of the policy to the request; the protocol is the only system text.

## v7 changes (vs v6_ptu)
- Header: <operating_policy> is binding operator policy, not requester input.
- Verbatim quoting of rules that fix a value, format, template or recipient.
- Policy-required writes are additional; merge only identical writes.
- Copy templates character for character; set included values explicitly.
- Audit: split the user request into separate asks, each with a matching write.
- Reminder in user_suffix states the policy is binding.
