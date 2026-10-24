"""Compare attempt groups: python scripts/opt/compare.py <domain> <train|test|all> <attemptA>... -- <attemptB>...

Prints mean pass rate of each group and tasks that flipped (majority-fail in A -> majority-pass in B and vice versa).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from domains import get  # noqa: E402

dom = get(sys.argv[1])
key = sys.argv[2]
split = json.load(open(dom["split"]))
ids = set(split["train"] + split["test"]) if key == "all" else set(split[key])
rest = sys.argv[3:]
i = rest.index("--")
A, B = rest[:i], rest[i + 1:]


def load(attempts):
    c = {t: 0 for t in ids}
    for a in attempts:
        for f in glob.glob(f"{dom['results']}/{a}/run_1/*.json"):
            j = json.load(open(f))
            t = j["benchmark_config"]["task_id"]
            if t in ids and j["statistics"].get("pass_at_1") == 1.0:
                c[t] += 1
    return c


a, b = load(A), load(B)
print(f"{key}: A={100 * sum(a.values()) / len(A) / len(ids):.1f}%  B={100 * sum(b.values()) / len(B) / len(ids):.1f}%")
if key == "train":  # never list individual test tasks
    print("gained:", sorted(t for t in ids if a[t] / len(A) < 0.5 <= b[t] / len(B)))
    print("lost:  ", sorted(t for t in ids if b[t] / len(B) < 0.5 <= a[t] / len(A)))
