"""Create a stratified train/test split: python scripts/opt/make_split.py <domain> [train_frac=0.333] [seed=7]

Stratifies by task creator hash (5th '_' field of task_id). Refuses to overwrite an existing split.
"""
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from domains import get  # noqa: E402

dom = get(sys.argv[1])
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 1 / 3
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
if os.path.exists(dom["split"]):
    sys.exit(f"{dom['split']} exists; delete it deliberately if you really want a new split")

groups = collections.defaultdict(list)
for f in sorted(glob.glob(f"{dom['configs']}/*.json")):
    tid = json.load(open(f))["task_id"]
    groups[tid.split("_")[4]].append(tid)
train, test = [], []
for k in sorted(groups):
    ids = groups[k]
    random.shuffle(ids)
    m = round(len(ids) * frac)
    train += ids[:m]
    test += ids[m:]
os.makedirs(dom["ws"], exist_ok=True)
for sub in ("overlays", "attempts"):
    os.makedirs(dom[sub], exist_ok=True)
json.dump({"train": train, "test": test}, open(dom["split"], "w"), indent=1)
print(f"{dom['split']}: {len(train)} train / {len(test)} test across {len(groups)} creators")
