"""Score a results folder against a task list: python score.py <results_run_dir> [split.json key]"""
import glob
import json
import os
import sys

run_dir = sys.argv[1]
ids = None
if len(sys.argv) > 3:
    ids = set(json.load(open(sys.argv[2]))[sys.argv[3]])

passed, failed, missing = [], [], []
seen = set()
for f in glob.glob(os.path.join(run_dir, "*.json")):
    d = json.load(open(f))
    tid = d["benchmark_config"]["task_id"]
    if ids is not None and tid not in ids:
        continue
    seen.add(tid)
    s = d.get("statistics", {})
    (passed if s.get("pass_at_1", 0) == 1.0 else failed).append(tid)
if ids is not None:
    missing = sorted(ids - seen)
n = len(passed) + len(failed) + len(missing)
print(f"{run_dir}: {len(passed)}/{n} = {100*len(passed)/max(n,1):.1f}%  (failed {len(failed)}, missing {len(missing)})")
if "-v" in sys.argv:
    print("FAILED:", *sorted(failed), sep="\n  ")
