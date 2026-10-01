"""DataSea wrapper backend.

    uv run uvicorn datasea.server:app --port 8700

Browser -> this backend -> EnterpriseOps MCP server(s) -> seeded DB.
Every tool call is proxied and logged here; the browser never talks to MCP directly.
Worker endpoints never return verifiers, seed files, auth tokens, or gold data.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import DATASEA_DIR
from .env import TaskEnvironment
from .export import export as run_export
from .provenance import repo_info
from .store import ERROR, FAILED, FLAGGED, IN_PROGRESS, PASSED, Store, now
from .tasks import load_catalog, worker_view

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("datasea")

app = FastAPI(title="DataSea demo collector")
store = Store()
live: Dict[str, TaskEnvironment] = {}
_locks: Dict[str, asyncio.Lock] = {}

STATIC = os.path.join(DATASEA_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class StartReq(BaseModel):
    worker_id: str
    task_id: str


class CallReq(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class FinishReq(BaseModel):
    outcome: str  # "submit" | "unclear"
    final_response: str = ""
    worker_note: str = ""


def _live(session_id: str) -> TaskEnvironment:
    env = live.get(session_id)
    if env is None:
        raise HTTPException(404, "No live environment for this session (already finished or server restarted)")
    return env


@app.get("/")
def worker_page():
    return FileResponse(os.path.join(STATIC, "worker.html"))


@app.get("/admin")
def admin_page():
    return FileResponse(os.path.join(STATIC, "admin.html"))


# ---------------------------------------------------------------- worker API

@app.get("/api/tasks")
def list_tasks():
    return [{"task_id": e["task_id"], "domain": e["domain"], "preview": e["config"]["user_prompt"][:140]}
            for e in load_catalog().values()]


@app.post("/api/sessions")
async def start_session(req: StartReq):
    worker_id = req.worker_id.strip()
    if not worker_id:
        raise HTTPException(400, "worker_id required")
    entry = load_catalog().get(req.task_id)
    if not entry:
        raise HTTPException(404, "Unknown task")
    env = TaskEnvironment(entry["config"])
    try:
        await env.start()
    except Exception as e:
        await env.reset()
        logger.exception("environment start failed")
        raise HTTPException(502, f"Could not start environment: {e}")

    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    view = worker_view(entry)
    store.create_session(
        {
            "session_id": session_id,
            "worker_id": worker_id,
            "task_id": entry["task_id"],
            "domain": entry["domain"],
            "pilot_only": entry["pilot_only"],
            "system_prompt": view["policy"],
            "user_prompt": view["instruction"],
            "selected_tools": entry["config"].get("selected_tools") or [],
            "available_tools": env.tools,
            "environment": env.environment_info,
            "provenance": {**repo_info(), "task_source": entry["source"], "task_mode": entry["mode"]},
        }
    )
    live[session_id] = env
    _locks[session_id] = asyncio.Lock()
    return {"session_id": session_id, "task": view, "tools": env.tools}


@app.post("/api/sessions/{session_id}/call")
async def call_tool(session_id: str, req: CallReq):
    env = _live(session_id)
    async with _locks[session_id]:
        ts = now()
        t0 = time.monotonic()
        try:
            res = await env.call_tool(req.tool_name, req.arguments)
            error = None
            if not res.get("success"):
                error = str(res.get("error"))
            elif res.get("error"):
                error = json_safe(res["error"])
            elif isinstance(res.get("result"), dict) and res["result"].get("isError"):
                error = _content_text(res["result"]) or "tool reported isError"
        except Exception as e:
            res, error = {"success": False, "error": str(e)}, str(e)
        ms = int((time.monotonic() - t0) * 1000)
        step = store.add_step(session_id, req.tool_name, req.arguments, res, error, ms, ts)
    return {"step": step, "timestamp": ts, "tool_name": req.tool_name, "arguments": req.arguments,
            "result": res, "display": _content_text(res.get("result")) if res.get("result") else None,
            "error": error, "duration_ms": ms}


@app.get("/api/sessions/{session_id}/history")
def history(session_id: str):
    return store.get_steps(session_id)


@app.post("/api/sessions/{session_id}/finish")
async def finish(session_id: str, req: FinishReq):
    env = _live(session_id)
    async with _locks[session_id]:
        steps = store.get_steps(session_id)
        tool_results = [{"tool_name": s["tool_name"], "arguments": s["arguments"]} for s in steps]
        try:
            verifier = await env.verify(req.final_response, tool_results)
            if req.outcome == "unclear":
                status = FLAGGED
            else:
                status = PASSED if verifier["overall_success"] else FAILED
        except Exception as e:
            logger.exception("verifier failed")
            verifier, status = {"overall_success": False, "error": str(e)}, ERROR
        store.finish_session(session_id, status, req.final_response, req.worker_note, verifier)
        await env.reset()
        store.mark_reset(session_id)
        live.pop(session_id, None)
    # Workers only learn that the session was recorded, not whether it passed.
    return {"session_id": session_id, "recorded": True}


# ---------------------------------------------------------------- admin API (localhost pilot only)

@app.get("/api/admin/sessions")
def admin_sessions():
    rows = store.list_sessions()
    for r in rows:
        s = store.get_session(r["session_id"])
        r["duration_seconds"] = (
            None if not s["ended_at"] else
            round((_iso(s["ended_at"]) - _iso(s["started_at"])), 1)
        )
        r["live"] = r["session_id"] in live
    return rows


@app.get("/api/admin/sessions/{session_id}")
def admin_session(session_id: str):
    s = store.get_session(session_id)
    if not s:
        raise HTTPException(404)
    return {"session": s, "steps": store.get_steps(session_id)}


@app.post("/api/admin/sessions/{session_id}/reset")
async def admin_reset(session_id: str):
    """Abandon a live session: keep its trajectory, drop its database."""
    env = _live(session_id)
    async with _locks[session_id]:
        store.finish_session(session_id, ERROR, "", "abandoned via admin reset", None)
        await env.reset()
        store.mark_reset(session_id)
        live.pop(session_id, None)
    return {"reset": True}


@app.post("/api/admin/export")
def admin_export(include_pilot: bool = True):
    from . import RUNTIME_DIR
    out = os.path.join(RUNTIME_DIR, "exports")
    return {"out_dir": out, "counts": run_export(out, include_pilot)}


@app.on_event("shutdown")
async def _cleanup():
    for sid, env in list(live.items()):
        store.finish_session(sid, ERROR, "", "server shutdown before submit", None)
        await env.reset()
        store.mark_reset(sid)


# ---------------------------------------------------------------- helpers

def _iso(s: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def json_safe(v: Any) -> str:
    import json
    return v if isinstance(v, str) else json.dumps(v)


def _content_text(result: Any) -> Optional[str]:
    """Pull the human-readable text out of an MCP tool result."""
    if not isinstance(result, dict):
        return None
    parts = [c.get("text", "") for c in result.get("content", []) if isinstance(c, dict) and c.get("type") == "text"]
    return "\n".join(parts) if parts else None
