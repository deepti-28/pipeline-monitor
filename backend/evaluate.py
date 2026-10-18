"""Runs all experiments on the real daily files: python -m backend.evaluate

Reads data/daily/*.parquet (NYC TLC Yellow Taxi trips, one file per day).
Each case = 14 consecutive real days as history plus one test day. With N days there are
N-14 windows; each window yields one faulty case (fault injected into the test day, fault
types cycled) and one clean case (test day untouched).
Writes results/eval_results.json and results/results.md.
"""
import json
import math

import numpy as np
import pandas as pd

from .agent.agent import run_agent
from .agent.baselines import alert_text_only, history_only
from .agent.tools import Ctx
from .config import FAULTS, RESULTS, ROOT, WINDOW
from .detector import detect_fixed, detect_robust, detect_rolling_z
from .inject_faults import inject
from .metrics import compute_snapshot

DAILY = ROOT / "data" / "daily"

# Documented calendar events, taken from the calendar and weather records, NOT from detector
# output: US federal holidays and the NYC winter storm of Jan 25-26, 2026.
EVENT_DATES = {
    "2025-12-25": "Christmas Day",
    "2026-01-01": "New Year's Day",
    "2026-01-19": "Martin Luther King Jr. Day",
    "2026-01-25": "NYC winter storm",
    "2026-01-26": "NYC winter storm",
}

DETECTORS = {
    "Fixed threshold": detect_fixed,
    "Rolling z-score": detect_rolling_z,
    "Median/MAD + KS/PSI (ours)": detect_robust,
}
OURS = "Median/MAD + KS/PSI (ours)"
AGENTS = ["Scripted baseline: alert text only", "Scripted baseline: metric history", "Full tool-using agent"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 3), round((c + m) / d, 3)]


def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def load_real_days():
    files = sorted(DAILY.glob("*.parquet"))
    if len(files) < WINDOW + FAULTS.__len__():
        raise RuntimeError(f"Need at least {WINDOW + len(FAULTS)} daily files in {DAILY}, found {len(files)}")
    dates, dfs, snaps = [], [], []
    for f in files:
        df = pd.read_parquet(f)
        dates.append(f.stem)                      # file name is the date, e.g. 2026-01-15
        dfs.append(df)
        snaps.append(compute_snapshot(df, f.stem))
    gaps = pd.to_datetime(pd.Series(dates)).diff().dt.days.dropna()
    if (gaps > 1).any():
        print("WARNING: missing days between daily files; some windows are not consecutive.")
    return dates, dfs, snaps


def run_eval(seed=7):
    dates, dfs, snaps = load_real_days()
    n_windows = len(dfs) - WINDOW

    cases = []
    for w in range(n_windows):
        hist = snaps[w:w + WINDOW]
        ref = pd.concat([x.sample(min(400, len(x)), random_state=w) for x in dfs[w:w + WINDOW]])
        day = dates[w + WINDOW]
        fault = FAULTS[w % len(FAULTS)]
        fdf, info = inject(dfs[w + WINDOW], fault, np.random.default_rng(1000 + w))

        for label in ("faulty", "clean"):
            df = dfs[w + WINDOW] if label == "clean" else fdf
            snap = snaps[w + WINDOW] if label == "clean" else compute_snapshot(df, day)
            case = {"id": f"{label}-{w}", "label": label, "date": day,
                    "fault": fault if label == "faulty" else None,
                    "info": info if label == "faulty" else None, "detected": {}}
            alerts = {}
            for name, fn in DETECTORS.items():
                a = fn(hist, snap, today_df=df, ref_df=ref)
                alerts[name] = a
                case["detected"][name] = a["is_anomaly"]
            if label == "faulty" and alerts[OURS]["is_anomaly"]:
                a = alerts[OURS]
                ctx = Ctx(hist=hist, today=snap, today_df=df, ref_df=ref, hist_all=hist)
                pred_text, calls_text = alert_text_only(a)
                pred_hist, calls_hist = history_only(ctx, a)
                out, trace = run_agent(ctx, a)
                case["pred"] = {AGENTS[0]: pred_text, AGENTS[1]: pred_hist, AGENTS[2]: out["cause"]}
                case["tool_calls"] = {AGENTS[0]: calls_text, AGENTS[1]: calls_hist, AGENTS[2]: len(trace)}
            cases.append(case)

    faulty = [c for c in cases if c["label"] == "faulty"]
    clean = [c for c in cases if c["label"] == "clean"]
    ordinary = [c for c in clean if c["date"] not in EVENT_DATES]

    detection = []
    for name in DETECTORS:
        per_fault = {f: _rate(sum(c["detected"][name] for c in faulty if c["fault"] == f),
                              sum(1 for c in faulty if c["fault"] == f)) for f in FAULTS}
        detection.append({
            "name": name,
            "detection": _rate(sum(c["detected"][name] for c in faulty), len(faulty)),
            "false_alarms": _rate(sum(c["detected"][name] for c in clean), len(clean)),
            "false_alarms_ordinary": _rate(sum(c["detected"][name] for c in ordinary), len(ordinary)),
            "flagged_clean_days": [c["date"] for c in clean if c["detected"][name]],
            "per_fault": per_fault,
        })

    flagged = [c for c in faulty if "pred" in c]
    agent = []
    for name in AGENTS:
        per_fault = {}
        for f in FAULTS:
            fc = [c for c in flagged if c["fault"] == f]
            per_fault[f] = _rate(sum(c["pred"][name] == f for c in fc), len(fc))
        correct = sum(c["pred"][name] == c["fault"] for c in flagged)
        agent.append({
            "name": name,
            "cause_accuracy_given_alert": _rate(correct, len(flagged)),
            "end_to_end_accuracy": _rate(correct, len(faulty)),
            "avg_tool_calls": round(float(np.mean([c["tool_calls"][name] for c in flagged])), 2) if flagged else None,
            "per_fault": per_fault,
        })

    return {
        "meta": {"n_faulty": len(faulty), "n_clean": len(clean), "n_ordinary_clean": len(ordinary),
                 "history_days": WINDOW, "seed": seed, "n_days": len(dfs),
                 "date_range": [dates[0], dates[-1]],
                 "data": "NYC TLC Yellow Taxi trips (real daily files)",
                 "faults": FAULTS, "event_dates": EVENT_DATES,
                 "note": "Faults are injected into real days; every window is used once for a faulty case "
                         "and once for a clean case. 'Clean' means no fault was injected, not that the day is "
                         "free of real anomalies. 'Ordinary' clean days exclude documented events (federal "
                         "holidays and the Jan 25-26 2026 NYC winter storm). The scripted baselines see less "
                         "information than the full agent; they are not LLM calls."},
        "detection": detection,
        "agent": agent,
        "cases": cases,
    }


def to_markdown(res) -> str:
    def pct(r):
        return "n/a" if r["rate"] is None else f"{r['rate']:.0%} ({r['k']}/{r['n']})"

    m = res["meta"]
    lines = [f"## Detection ({m['n_faulty']} faulty days, {m['n_clean']} clean days; "
             f"{m['date_range'][0]} to {m['date_range'][1]})", "",
             f"| Detector | Detection rate | False alarms, all clean days | "
             f"False alarms, ordinary days ({m['n_ordinary_clean']}) |", "|---|---|---|---|"]
    for d in res["detection"]:
        lines.append(f"| {d['name']} | {pct(d['detection'])} | {pct(d['false_alarms'])} | "
                     f"{pct(d['false_alarms_ordinary'])} |")
    lines += ["", "Documented events excluded from 'ordinary': " +
              ", ".join(f"{k} ({v})" for k, v in m["event_dates"].items()), "",
              "Clean days flagged by each detector:"]
    for d in res["detection"]:
        lines.append(f"- {d['name']}: {', '.join(d['flagged_clean_days']) or 'none'}")
    lines += ["", "## Root-cause accuracy (on days our detector flagged)", "",
              f"| Setup | Cause accuracy | End-to-end (all {m['n_faulty']} faulty) | Avg tool calls |", "|---|---|---|---|"]
    for a in res["agent"]:
        lines.append(f"| {a['name']} | {pct(a['cause_accuracy_given_alert'])} | {pct(a['end_to_end_accuracy'])} "
                     f"| {a['avg_tool_calls']} |")
    lines += ["", "## Accuracy per fault type (cause accuracy)", "",
              "| Fault | " + " | ".join(a["name"] for a in res["agent"]) + " |",
              "|---|" + "---|" * len(res["agent"])]
    for f in FAULTS:
        lines.append(f"| {f} | " + " | ".join(pct(a["per_fault"][f]) for a in res["agent"]) + " |")
    lines += ["", "## Detection rate per fault type", "",
              "| Fault | " + " | ".join(d["name"] for d in res["detection"]) + " |",
              "|---|" + "---|" * len(res["detection"])]
    for f in FAULTS:
        lines.append(f"| {f} | " + " | ".join(pct(d["per_fault"][f]) for d in res["detection"]) + " |")
    return "\n".join(lines) + "\n"


def save(res):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval_results.json").write_text(json.dumps(res, indent=1))
    (RESULTS / "results.md").write_text(to_markdown(res))
    from . import db
    db.clear("eval_results")
    db.insert("eval_results", {k: res[k] for k in ("meta", "detection", "agent")})


if __name__ == "__main__":
    r = run_eval()
    save(r)
    print(to_markdown(r))