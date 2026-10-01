"""Pilot task catalog.

Tasks are imported from the official HuggingFace dataset at a pinned revision
and converted into the exact config dicts that upstream `evaluate.py` writes
before calling `load_config`. Every imported public task is marked
`pilot_only = true` so it can never be counted as untouched evaluation data.

Usage:
    python -m datasea.tasks import --domain email --task_id <id> [--task_id <id> ...]
    python -m datasea.tasks list
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from . import DATASEA_DIR

HF_DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
CATALOG_PATH = os.path.join(DATASEA_DIR, "tasks", "pilot_tasks.jsonl")

# Same conversion as evaluate.py (--hf_dataset branch).
_JSON_STRING_FIELDS = {"gym_servers_config", "verifiers"}
_HF_ONLY_FIELDS = {"task_id", "domain"}


def _hf_revision_sha(dataset: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset, revision=revision).sha


def import_tasks(domain: str, task_ids: List[str], mode: str = "oracle", revision: str = "main") -> List[Dict[str, Any]]:
    from datasets import load_dataset

    sha = _hf_revision_sha(HF_DATASET, revision)
    ds = load_dataset(HF_DATASET, mode, split=domain, revision=sha)
    wanted = set(task_ids)
    entries = []
    for row in ds:
        if wanted and row["task_id"] not in wanted:
            continue
        config = {}
        for k, v in row.items():
            if k in _HF_ONLY_FIELDS:
                continue
            if k in _JSON_STRING_FIELDS and isinstance(v, str):
                v = json.loads(v)
            config[k] = v
        entries.append(
            {
                "task_id": row["task_id"],
                "domain": row["domain"],
                "mode": mode,
                "pilot_only": True,
                "source": {"type": "official_public", "hf_dataset": HF_DATASET, "hf_revision": sha, "hf_config": mode},
                "config": config,
            }
        )
    missing = wanted - {e["task_id"] for e in entries}
    if missing:
        raise SystemExit(f"Task ids not found in {domain}/{mode}: {sorted(missing)}")
    return entries


def load_catalog() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(CATALOG_PATH):
        return {}
    with open(CATALOG_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return {e["task_id"]: e for e in entries}


def save_catalog(catalog: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        for e in catalog.values():
            f.write(json.dumps(e) + "\n")


def worker_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Only what an agent under evaluation would see: policy + request. No verifiers, seeds, or tokens."""
    cfg = entry["config"]
    return {
        "task_id": entry["task_id"],
        "domain": entry["domain"],
        "instruction": cfg["user_prompt"],
        "policy": cfg["system_prompt"],
    }


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return load_catalog().get(task_id)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--domain", required=True)
    imp.add_argument("--task_id", action="append", default=[])
    imp.add_argument("--mode", default="oracle")
    imp.add_argument("--revision", default="main")
    sub.add_parser("list")
    args = p.parse_args()

    catalog = load_catalog()
    if args.cmd == "import":
        for e in import_tasks(args.domain, args.task_id, args.mode, args.revision):
            catalog[e["task_id"]] = e
            print(f"imported {e['domain']}/{e['task_id']} (pilot_only=true, rev {e['source']['hf_revision'][:10]})")
        save_catalog(catalog)
    else:
        for e in catalog.values():
            print(e["domain"], e["task_id"], "pilot_only=" + str(e["pilot_only"]).lower())


if __name__ == "__main__":
    main()
