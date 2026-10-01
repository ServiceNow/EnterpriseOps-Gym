"""SQLite persistence for sessions and trajectory steps. Nothing is ever deleted."""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import RUNTIME_DIR

DB_PATH = os.path.join(RUNTIME_DIR, "datasea.sqlite")

# Session status values.
IN_PROGRESS = "in_progress"
PASSED = "passed"
FAILED = "failed"
FLAGGED = "flagged_unclear"
ERROR = "error"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    pilot_only INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    selected_tools_json TEXT NOT NULL,
    available_tools_json TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    final_response TEXT,
    worker_note TEXT,
    verifier_json TEXT,
    verifier_pass INTEGER,
    reset_at TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    duration_ms INTEGER,
    PRIMARY KEY (session_id, step)
);
"""

_JSON_COLS = {
    "selected_tools_json": "selected_tools",
    "available_tools_json": "available_tools",
    "environment_json": "environment",
    "provenance_json": "provenance",
    "verifier_json": "verifier",
    "arguments_json": "arguments",
    "result_json": "result",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def _exec(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    @staticmethod
    def _decode(row: sqlite3.Row) -> Dict[str, Any]:
        out = {}
        for k in row.keys():
            v = row[k]
            if k in _JSON_COLS:
                out[_JSON_COLS[k]] = json.loads(v) if v is not None else None
            else:
                out[k] = v
        return out

    def create_session(self, s: Dict[str, Any]) -> None:
        self._exec(
            """INSERT INTO sessions (session_id, worker_id, task_id, domain, pilot_only, status, started_at,
               system_prompt, user_prompt, selected_tools_json, available_tools_json, environment_json, provenance_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                s["session_id"], s["worker_id"], s["task_id"], s["domain"], int(s["pilot_only"]), IN_PROGRESS,
                now(), s["system_prompt"], s["user_prompt"], json.dumps(s["selected_tools"]),
                json.dumps(s["available_tools"]), json.dumps(s["environment"]), json.dumps(s["provenance"]),
            ),
        )

    def add_step(self, session_id: str, tool_name: str, arguments: Dict[str, Any], result: Any,
                 error: Optional[str], duration_ms: int, timestamp: str) -> int:
        with self._lock:
            n = self._conn.execute("SELECT COALESCE(MAX(step), 0) FROM steps WHERE session_id=?", (session_id,)).fetchone()[0] + 1
            self._conn.execute(
                "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?)",
                (session_id, n, timestamp, tool_name, json.dumps(arguments), json.dumps(result), error, duration_ms),
            )
            self._conn.commit()
        return n

    def finish_session(self, session_id: str, status: str, final_response: str, worker_note: str,
                       verifier: Optional[Dict[str, Any]]) -> None:
        self._exec(
            "UPDATE sessions SET status=?, ended_at=?, final_response=?, worker_note=?, verifier_json=?, verifier_pass=? WHERE session_id=?",
            (
                status, now(), final_response, worker_note, json.dumps(verifier) if verifier is not None else None,
                None if verifier is None else int(bool(verifier.get("overall_success"))), session_id,
            ),
        )

    def mark_reset(self, session_id: str) -> None:
        self._exec("UPDATE sessions SET reset_at=? WHERE session_id=?", (now(), session_id))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return self._decode(row) if row else None

    def get_steps(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM steps WHERE session_id=? ORDER BY step", (session_id,)).fetchall()
        return [self._decode(r) for r in rows]

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.session_id, s.worker_id, s.task_id, s.domain, s.pilot_only, s.status, s.started_at,
                          s.ended_at, s.verifier_pass, s.reset_at, s.worker_note,
                          (SELECT COUNT(*) FROM steps t WHERE t.session_id = s.session_id) AS num_tool_calls
                   FROM sessions s ORDER BY s.started_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]
