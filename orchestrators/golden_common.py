"""Shared golden-file resolution for the golden orchestrators.

Both :class:`GoldenOrchestrator` (deterministic, no-LLM replay) and
:class:`GoldenGuidedOrchestrator` (trajectory-conditioned ReAct) need to locate
and load the raw task file that carries the golden trajectory
(``executions.Oracle.config.manual_tool_executions``). The processed configs
consumed by the executor do NOT contain that trajectory, so the raw file is
resolved separately from ``self.config_path``.

This logic used to live inline in ``golden.py``. It is factored out here as a
mixin so both orchestrators share exactly one implementation. The behaviour is
identical to the original ``GoldenOrchestrator`` methods.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Root directory holding the raw task files (the ones that still carry
# ``executions.Oracle.config.manual_tool_executions``). The processed configs
# consumed by the executor do NOT contain the golden trajectory, so we resolve
# the raw file separately. Override with the GOLDEN_RAW_DATA_ROOT env var.
RAW_DATA_ROOT = os.environ.get(
    "GOLDEN_RAW_DATA_ROOT",
    "/Users/shivakrishnareddy.ma/Documents/git/EnterpriseOps-Gym (Backup)/"
    "EnterpriseOps-Gym copy/data/all",
)

# Static template resolved against RAW_DATA_ROOT with {domain} and {task_id}
# substituted, i.e. <RAW_DATA_ROOT>/<domain>/<task_id>.json
RAW_PATH_TEMPLATE = "{root}/{domain}/{task_id}.json"

# Static final response for deterministic golden runs (no LLM => no answer).
GOLDEN_FINAL_RESPONSE = "Empty - Manual Golden Run"

# Known domains, used to infer the domain from a local config path.
KNOWN_DOMAINS = {
    "csm", "itsm", "hr", "email", "calendar", "drive", "teams", "hybrid",
}


class GoldenFileMixin:
    """Resolve and load the golden ``manual_tool_executions`` for a task.

    Expects ``self.config_path`` to be set by the host orchestrator's
    ``__init__`` (the path to the processed config the executor was given).
    """

    def _resolve_domain_and_task_id(self) -> Tuple[str, str]:
        """Derive (domain, task_id) from the config file path.

        Handles two config naming conventions:
        - HuggingFace export: ``<mode>__<domain>__<idx>__<task_id>.json``
        - Local files: ``.../<domain>/[processed_*/]<task_id>.json`` where the
          filename base is the task_id and the domain is an ancestor dir.
        """
        if not self.config_path:
            raise ValueError(
                f"{type(self).__name__} requires 'config_path' to locate the raw "
                "task file, but none was provided."
            )

        base = os.path.basename(self.config_path)
        stem = base[:-5] if base.endswith(".json") else base

        # HuggingFace export naming: mode__domain__idx__task_id
        if "__" in stem:
            parts = stem.split("__")
            if len(parts) >= 4:
                domain = parts[1]
                task_id = "__".join(parts[3:])
                return domain, task_id

        # Local naming: task_id is the filename stem, domain is an ancestor dir.
        task_id = stem
        abs_path = os.path.abspath(self.config_path)
        path_parts = abs_path.split(os.sep)

        domain = None
        # Walk the ancestor directories (excluding the filename) looking for a
        # known domain segment, closest to the file first.
        for part in reversed(path_parts[:-1]):
            if part in KNOWN_DOMAINS:
                domain = part
                break

        if domain is None:
            raise ValueError(
                f"Could not determine domain for config path '{self.config_path}'. "
                f"Expected a known domain segment in the path or an HF-style "
                f"filename '<mode>__<domain>__<idx>__<task_id>.json'."
            )

        return domain, task_id

    def _load_manual_tool_executions(self) -> List[Dict[str, Any]]:
        """Locate and load the golden manual_tool_executions for this task."""
        domain, task_id = self._resolve_domain_and_task_id()
        raw_path = RAW_PATH_TEMPLATE.format(
            root=RAW_DATA_ROOT, domain=domain, task_id=task_id
        )

        logger.info(f"[GOLDEN] Loading raw task file: {raw_path}")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw task file not found for golden replay: {raw_path} "
                f"(domain={domain}, task_id={task_id}). "
                f"Set GOLDEN_RAW_DATA_ROOT to point at the raw data/all dir."
            )

        with open(raw_path, "r", encoding="utf-8") as f:
            golden_task = json.load(f)

        try:
            manual_tool_executions = golden_task["executions"]["Oracle"]["config"][
                "manual_tool_executions"
            ]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Raw task file '{raw_path}' does not contain "
                f"executions.Oracle.config.manual_tool_executions: {e}"
            )

        if not isinstance(manual_tool_executions, list):
            raise ValueError(
                f"manual_tool_executions in '{raw_path}' is not a list "
                f"(got {type(manual_tool_executions).__name__})."
            )

        logger.info(
            f"[GOLDEN] Loaded {len(manual_tool_executions)} golden tool "
            f"execution(s) for task {task_id} (domain={domain})"
        )
        return manual_tool_executions
