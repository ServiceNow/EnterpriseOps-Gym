"""Reproducibility metadata: repo commit, dataset revision, docker image digests."""

import functools
import hashlib
import json
import subprocess
from typing import Any, Dict, Optional

from . import DATASEA_VERSION, REPO_ROOT


def _run(cmd) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def repo_info() -> Dict[str, Any]:
    return {
        "repo_commit": _run(["git", "rev-parse", "HEAD"]),
        "repo_upstream_url": _run(["git", "remote", "get-url", "upstream"]),
        "repo_origin_url": _run(["git", "remote", "get-url", "origin"]),
        "repo_dirty_files": (_run(["git", "status", "--porcelain"]) or "").splitlines(),
        "datasea_version": DATASEA_VERSION,
    }


@functools.lru_cache(maxsize=32)
def docker_image_for_port(port: int) -> Dict[str, Any]:
    """Best-effort lookup of the container image serving a given host port."""
    raw = _run(["docker", "ps", "--format", "{{json .}}"])
    if not raw:
        return {}
    for line in raw.splitlines():
        info = json.loads(line)
        if f":{port}->" in info.get("Ports", ""):
            image = info.get("Image")
            digest = _run(["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"])
            return {"image": image, "image_digest": digest, "container_id": info.get("ID")}
    return {}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
