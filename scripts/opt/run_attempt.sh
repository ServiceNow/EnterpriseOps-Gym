#!/bin/bash
# usage: scripts/opt/run_attempt.sh <domain> <attempt_name> [overlay.json]
# Runs ALL tasks of the domain (react, concurrency 4) with the overlay, then archives prompt + scores.
set -u
DOM=$1; NAME=$2; OV=${3:-}
P() { .venv/bin/python scripts/opt/domains.py "$DOM" "$1"; }
CONFIGS=$(P configs); RES=$(P results); LOGS=$(P logs)
LLM=${OPT_LLM_CONFIG:-conf/llm/gpt_5p6_terra_medium.json}
mkdir -p "$RES" "$LOGS"
if [ -n "$OV" ]; then export PROMPT_OVERLAY=$(realpath "$OV"); else unset PROMPT_OVERLAY; fi
.venv/bin/python evaluate.py --configs_folder "$CONFIGS" --llm_config "$LLM" \
  --output_folder "$RES/$NAME" --orchestrator react --concurrency "${OPT_CONCURRENCY:-4}" --num_runs 1 \
  > "$LOGS/$NAME.log" 2>&1
.venv/bin/python scripts/opt/archive_attempt.py "$DOM" "$NAME" "${OV:-none}"
