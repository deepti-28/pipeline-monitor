"""Runs all experiments: python -m backend.evaluate

60 cases = 30 faulty (5 per fault type) + 30 clean. Each case is 14 clean history days
plus one test day. Writes results/eval_results.json and results/results.md.
"""
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .agent.agent import run_agent
from .agent.baselines import alert_text_only, history_only
from .agent.tools import Ctx
from .config import FAULTS, RESULTS, WINDOW
from .detector import detect_fixed, detect_robust, detect_rolling_z
from .inject_faults import inject
from .metrics import compute_snapshot
from .synth import make_day

DETECTORS = {
    "Fixed threshold": detect_fixed,
    "Rolling z-score": detect_rolling_z,
    "Median/MAD + KS/PSI (ours)": detect_robust,
}
OURS = "Median/MAD + KS/PSI (ours)"
AGENTS = ["LLM-style: alert text only", "LLM-style: metric history", "Full tool-using agent"]


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


def run_eval(n_pool=45, seed=7, n_cases=30):
    rng = np.random.default_rng(seed)
    start = date(2024, 6, 3)
    dfs, snaps = [], []
    for i in range(n_pool):
        d = start + timedelta(days=i)
        df = make_day(d, rng)
        dfs.append(df)
        snaps.append(compute_snapshot(df, str(d)))

    cases = []
    for i in range(n_cases):
        hist = snaps[i:i + WINDOW]
        ref = pd.concat([x.sample(400, random_state=i) for x in dfs[i:i + WINDOW]])
        day = str(start + timedelta(days=i + WINDOW))
        fault = FAULTS[i % len(FAULTS)]
        fdf, info = inject(dfs[i + WINDOW], fault, np.random.default_rng(1000 + i))
        for label, df in (("clean", dfs[i + WINDOW]), ("faulty", fdf)):
            snap = snaps[i + WINDOW] if label == "clean" else compute_snapshot(df, day)
            case = {"id": f"{label}-{i}", "label": label, "fault": fault if label == "faulty" else None,
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

    detection = []
    for name in DETECTORS:
        per_fault = {f: _rate(sum(c["detected"][name] for c in faulty if c["fault"] == f),
                              sum(1 for c in faulty if c["fault"] == f)) for f in FAULTS}
        detection.append({
            "name": name,
            "detection": _rate(sum(c["detected"][name] for c in faulty), len(faulty)),
            "false_alarms": _rate(sum(c["detected"][name] for c in clean), len(clean)),
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
        "meta": {"n_faulty": len(faulty), "n_clean": len(clean), "history_days": WINDOW, "seed": seed,
                 "data": "synthetic NYC-taxi-shaped (see README)", "faults": FAULTS,
                 "note": "The two LLM-style rows are scripted analyzers that see less information than the "
                         "full agent; they are stand-ins for LLM setups, not LLM calls."},
        "detection": detection,
        "agent": agent,
        "cases": cases,
    }


def to_markdown(res) -> str:
    def pct(r):
        return "n/a" if r["rate"] is None else f"{r['rate']:.0%} ({r['k']}/{r['n']})"

    lines = ["## Detection (30 faulty days, 30 clean days)", "",
             "| Detector | Detection rate | False alarm rate |", "|---|---|---|"]
    for d in res["detection"]:
        lines.append(f"| {d['name']} | {pct(d['detection'])} | {pct(d['false_alarms'])} |")
    lines += ["", "## Root-cause accuracy (on days our detector flagged)", "",
              "| Setup | Cause accuracy | End-to-end (all 30 faulty) | Avg tool calls |", "|---|---|---|---|"]
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
