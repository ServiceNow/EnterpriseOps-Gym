"""Derive structural variants of an overlay whose instructions live in system_suffix.

python scripts/opt/make_variants.py <base.json> <prefix_out.json> <policy_to_user_out.json>
  prefix_out:          same instructions placed BEFORE the task system prompt
  policy_to_user_out:  instructions become the system message; task system prompt moves into the user turn
Review the outputs: wording such as "the policy above" may need adjusting for the new position.
"""
import json
import sys

base, prefix_out, ptu_out = sys.argv[1:4]
d = json.load(open(base))
instr = d["system_suffix"].strip()
rest = {k: v for k, v in d.items() if k not in ("system_suffix", "system_prefix", "policy_to_user")}

json.dump({**rest, "system_prefix": instr + "\n\n---\n\n"}, open(prefix_out, "w"), indent=1)
json.dump({**rest, "policy_to_user": True, "system_prefix": instr}, open(ptu_out, "w"), indent=1)
print("ok")
