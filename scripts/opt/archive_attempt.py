"""Archive one attempt: python scripts/opt/archive_attempt.py <domain> <name> <overlay.json|none> [analysis.md]

Heavy (untracked):  <results>/<name>/{run_1/, overlay.json, scores.txt, eval.log}
Light (tracked):    <ws>/attempts/<name>/{overlay.json, scores.txt[, analysis.md]}
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from domains import get  # noqa: E402

dom = get(sys.argv[1])
name, overlay = sys.argv[2], sys.argv[3]
analysis = sys.argv[4] if len(sys.argv) > 4 else None
res = f"{dom['results']}/{name}"
light = f"{dom['attempts']}/{name}"
os.makedirs(light, exist_ok=True)

lines = []
for key in ("train", "test", None):
    cmd = [".venv/bin/python", "scripts/opt/score.py", f"{res}/run_1"]
    if key:
        cmd += [dom["split"], key]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    lines.append(f"{key or 'all'}: {out}")
scores = "\n".join(lines) + "\n"
for d in (res, light):
    open(f"{d}/scores.txt", "w").write(scores)
    if overlay != "none":
        shutil.copy(overlay, f"{d}/overlay.json")
    else:
        open(f"{d}/overlay.json", "w").write("{}\n")
for log in (f"{dom['logs']}/{name}.log", f"logs/{name}.log"):
    if os.path.exists(log):
        shutil.copy(log, f"{res}/eval.log")
        break
if analysis:
    shutil.copy(analysis, f"{light}/analysis.md")
print(scores)
