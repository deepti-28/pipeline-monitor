"""Live pipeline: load a day -> metrics -> detect -> (agent) -> store alert.

Framework-free so it can be tested without FastAPI; main.py wraps it in HTTP.
"""
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import db
from .agent.agent import run_agent
from .agent.tools import Ctx
from .config import EXPECTED_COLS, FAULTS, WINDOW
from .detector import detect_robust
from .inject_faults import inject
from .metrics import compute_snapshot, conform, flatten
from .prepare_data import daily_files, ensure_daily_files, read_day

LOCK = threading.RLock()


def _state():
    s = db.get("state", 1)
    return s or {"cursor": 0, "pending_fault": None}


def _save_state(s):
    db.put("state", 1, s)


def load_day(df: pd.DataFrame, day: str):
    """Loader: append a day's rows to the SQLite `trips` table (missing columns become NULL)."""
    d = conform(df)[EXPECTED_COLS].copy()
    d["tpep_pickup_datetime"] = d["tpep_pickup_datetime"].astype(str)
    d["load_date"] = day
    with db.conn() as c:
        d.to_sql("trips", c, if_exists="append", index=False)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trips_load_date ON trips(load_date)")


def _ref_sample(dates, per_day=400):
    if not dates:
        return None
    frames = []
    with db.conn() as c:
        for d in dates:
            frames.append(pd.read_sql(
                "SELECT * FROM trips WHERE load_date=? ORDER BY RANDOM() LIMIT ?", c, params=(d, per_day)))
    return pd.concat(frames, ignore_index=True)


def ensure_ready():
    with LOCK:
        ensure_daily_files()
        if db.get("state", 1) is None or not db.all_docs("daily_metrics"):
            reset()


def reset():
    """Drop all live state and re-load the first WINDOW days as the warm-up baseline."""
    with LOCK:
        db.drop_trips()
        for coll in ("daily_metrics", "alerts", "agent_runs", "state"):
            db.clear(coll)
        files = daily_files()
        for i, path in enumerate(files[:WINDOW]):
            df = read_day(path)
            load_day(df, path.stem)
            snap = compute_snapshot(df, path.stem)
            snap.update(flagged=False, warmup=True, injected=None, alert_id=None)
            db.put("daily_metrics", i + 1, snap)
        _save_state({"cursor": min(WINDOW, len(files)), "pending_fault": None})


def status():
    s = _state()
    files = daily_files()
    return {"cursor": s["cursor"], "total_days": len(files), "warmup_days": WINDOW,
            "pending_fault": s["pending_fault"], "done": s["cursor"] >= len(files),
            "next_date": files[s["cursor"]].stem if s["cursor"] < len(files) else None,
            "faults": FAULTS}


def queue_fault(fault):
    with LOCK:
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}")
        s = _state()
        s["pending_fault"] = fault
        _save_state(s)
    return status()


def run_next_day():
    with LOCK:
        s = _state()
        files = daily_files()
        if s["cursor"] >= len(files):
            return {"done": True, "status": status()}
        idx = s["cursor"]
        path = files[idx]
        day = path.stem
        df = read_day(path)

        truth = None
        if s["pending_fault"]:
            df, truth = inject(df, s["pending_fault"], np.random.default_rng(1000 + idx))

        load_day(df, day)
        snap = compute_snapshot(df, day)

        all_snaps = db.all_docs("daily_metrics")
        clean = [m for m in all_snaps if not m["flagged"]][-WINDOW:]
        ref_df = _ref_sample([m["date"] for m in clean[-10:]])
        alert = detect_robust(clean, snap, today_df=conform(df), ref_df=ref_df)

        snap.update(flagged=alert["is_anomaly"], warmup=False, injected=truth, alert_id=None)
        alert_doc = None
        if alert["is_anomaly"]:
            ctx = Ctx(hist=clean, today=snap, today_df=conform(df), ref_df=ref_df, hist_all=all_snaps[-WINDOW:])
            out, trace = run_agent(ctx, alert)
            alert_doc = {
                "date": day, "status": "open", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "detector": alert["detector"], "text": alert["text"], "flags": alert["flags"],
                "cause": out["cause"], "confidence": out["confidence"], "evidence": out["evidence"],
                "suggested_fix": out["suggested_fix"], "tool_calls": len(trace),
                "injected": truth, "correct": (out["cause"] == truth["fault"]) if truth else None,
            }
            aid = db.insert("alerts", alert_doc)
            alert_doc["id"] = aid
            db.insert("agent_runs", {"alert_id": aid, "trace": trace})
            snap["alert_id"] = aid
        db.put("daily_metrics", idx + 1, snap)
        s["cursor"] = idx + 1
        s["pending_fault"] = None
        _save_state(s)
        return {"date": day, "anomaly": alert["is_anomaly"], "injected": truth, "alert": alert_doc,
                "status": status()}


# ------------------------------------------------------------------ read helpers
def features():
    docs = db.all_docs("daily_metrics")
    return sorted(flatten(docs[0]).keys()) if docs else []


def metric_series(feature):
    out = []
    for m in db.all_docs("daily_metrics"):
        v = flatten(m).get(feature)
        out.append({"date": m["date"], "value": v, "flagged": m["flagged"], "warmup": m.get("warmup", False),
                    "injected": (m.get("injected") or {}).get("fault"), "alert_id": m.get("alert_id")})
    return out


def list_alerts():
    return list(reversed(db.all_docs("alerts")))


def get_alert(alert_id):
    a = db.get("alerts", alert_id)
    if not a:
        return None
    runs = [r for r in db.all_docs("agent_runs") if r["alert_id"] == alert_id]
    a["trace"] = runs[0]["trace"] if runs else []
    return a


def set_alert_status(alert_id, new_status):
    if new_status not in ("open", "acknowledged", "resolved"):
        raise ValueError("status must be open, acknowledged or resolved")
    a = db.get("alerts", alert_id)
    if not a:
        return None
    a["status"] = new_status
    db.put("alerts", alert_id, a)
    return a
