"""FastAPI app: REST + WebSocket, and it serves the dashboard from /frontend.

Run:  uvicorn backend.main:app --reload
"""
import gc
import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline
from .config import FAULTS, RESULTS, ROOT

log = logging.getLogger("pipeline-monitor")

FRONTEND = ROOT / "frontend"
EVAL_PATH = RESULTS / "eval_results.json"

# Set RUN_EVAL_IF_MISSING=1 on your own machine only. Never on Render (too much memory).
RUN_EVAL_IF_MISSING = os.getenv("RUN_EVAL_IF_MISSING") == "1"

# Warm-up state: the server starts first, the data setup runs in the background.
state = {"ready": False, "error": None}


class FaultRequest(BaseModel):
    fault: str | None = None


class StatusRequest(BaseModel):
    status: str


class Hub:
    """Tracks open WebSocket clients and broadcasts new alerts."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


hub = Hub()


def _warm_up():
    """Runs in a background thread so the port opens right away."""
    try:
        pipeline.ensure_ready()
        state["ready"] = True
        log.info("Warm-up finished")
    except Exception as e:  # keep the server alive and show the error in /api/status
        state["error"] = str(e)
        log.exception("Warm-up failed")
    finally:
        gc.collect()


def _require_ready():
    if state["error"]:
        raise HTTPException(500, f"Startup failed: {state['error']}")
    if not state["ready"]:
        raise HTTPException(503, "Server is warming up, try again in a few seconds")


def _load_eval():
    """Read the pre-computed evaluation. It is generated locally with
    `python -m backend.evaluate` and committed to the repo."""
    if not EVAL_PATH.exists() and RUN_EVAL_IF_MISSING:
        from .evaluate import run_eval, save  # lazy import: heavy
        save(run_eval())
    if not EVAL_PATH.exists():
        raise HTTPException(
            404,
            "Evaluation results not found. Run `python -m backend.evaluate` "
            "locally and commit results/eval_results.json.",
        )
    return json.loads(EVAL_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Pipeline Health Monitor", lifespan=lifespan)


@app.get("/api/status")
def get_status():
    if state["error"]:
        return {"ready": False, "error": state["error"]}
    if not state["ready"]:
        return {"ready": False, "warming_up": True}
    return {"ready": True, **pipeline.status()}


@app.post("/run-day")
async def run_day():
    _require_ready()
    result = await run_in_threadpool(pipeline.run_next_day)
    if result.get("alert"):
        await hub.broadcast({"type": "alert", "alert": result["alert"]})
    await hub.broadcast({"type": "status", "status": result["status"]})
    return result


@app.post("/inject-fault")
def inject_fault(req: FaultRequest):
    _require_ready()
    if req.fault is not None and req.fault not in FAULTS:
        raise HTTPException(400, f"fault must be one of {FAULTS}")
    return pipeline.queue_fault(req.fault)


@app.post("/reset")
async def reset():
    _require_ready()
    await run_in_threadpool(pipeline.reset)
    await hub.broadcast({"type": "reset"})
    return pipeline.status()


@app.get("/features")
def get_features():
    _require_ready()
    return pipeline.features()


@app.get("/metrics")
def get_metrics(feature: str = Query("row_count", description="e.g. row_count, fare_amount.mean")):
    _require_ready()
    if feature not in pipeline.features():
        raise HTTPException(404, f"unknown feature {feature!r}")
    return {"feature": feature, "points": pipeline.metric_series(feature)}


@app.get("/alerts")
def get_alerts():
    _require_ready()
    return pipeline.list_alerts()


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int):
    _require_ready()
    a = pipeline.get_alert(alert_id)
    if not a:
        raise HTTPException(404, "alert not found")
    return a


@app.post("/alerts/{alert_id}/status")
def set_alert_status(alert_id: int, req: StatusRequest):
    _require_ready()
    try:
        a = pipeline.set_alert_status(alert_id, req.status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not a:
        raise HTTPException(404, "alert not found")
    return a


@app.get("/eval-results")
async def eval_results():
    res = await run_in_threadpool(_load_eval)
    res.pop("cases", None)  # keep the payload small; full cases stay in results/
    return res


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; clients only listen
    except WebSocketDisconnect:
        hub.disconnect(ws)


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")