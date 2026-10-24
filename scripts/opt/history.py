"""Train-task failure history across all attempts: python scripts/opt/history.py <domain> [attempt_to_focus]

Shows, for each train task failing in the focus attempt (default: latest), in how many attempts it failed.
Tasks failing in every attempt are likely benchmark defects or the hard core; flaky ones are noise-dominated.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from domains import get  # noqa: E402

dom = get(sys.argv[1])
train = set(json.load(open(dom["split"]))["train"])
attempts = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{dom['results']}/*/run_1"))
attempts.sort(key=lambda a: os.path.getmtime(f"{dom['results']}/{a}/run_1"))
focus = sys.argv[2] if len(sys.argv) > 2 else attempts[-1]


def fails(a):
    out = set()
    for f in glob.glob(f"{dom['results']}/{a}/run_1/*.json"):
        j = json.load(open(f))
        t = j["benchmark_config"]["task_id"]
        if t in train and j["statistics"].get("pass_at_1") != 1.0:
            out.add(t)
    return out


F = {a: fails(a) for a in attempts}
print(f"focus={focus}; attempts={len(attempts)}")
for t in sorted(F[focus], key=lambda t: -sum(t in F[a] for a in attempts)):
    print(f"  {t}  failed {sum(t in F[a] for a in attempts)}/{len(attempts)}")
