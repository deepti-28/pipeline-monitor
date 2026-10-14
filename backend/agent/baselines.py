"""The two weaker comparison setups for the agent evaluation.

These are scripted analyzers standing in for the 'LLM with alert text only' and 'LLM with
metric history' setups: they see strictly less information than the full agent.
Swap in real LLM calls here to reproduce the plan's original comparison.
"""
from ..detector import THR
from .tools import Ctx, get_metric_history


def alert_text_only(alert: dict):
    """Reads only the alert: top-scoring flag (value vs baseline) -> cause. 0 tool calls."""
    if not alert["flags"]:
        return "distribution_drift", 0
    top = max(alert["flags"], key=lambda f: abs(f["score"]))
    feat, kind = top["feature"], top["kind"]
    if kind == "schema":
        return "schema_change", 0
    if feat == "dup_rate":
        return "duplicate_rows", 0
    if feat == "row_count":
        return ("missing_partial_day" if top["value"] < top["baseline"] else "duplicate_rows"), 0
    if feat.endswith(".null_rate"):
        return "null_spike", 0
    return "distribution_drift", 0


def history_only(ctx: Ctx, alert: dict):
    """Sees metric history for the flagged features only (no schema, no raw rows, no tests)."""
    calls = 0

    def hist(feat):
        nonlocal calls
        calls += 1
        return get_metric_history(ctx, feat)

    feats = [f["feature"] for f in alert["flags"] if f["kind"] in ("z", "rule")]
    if "dup_rate" in feats and (hist("dup_rate")["today"] or 0) > 0.015:
        return "duplicate_rows", calls
    if "row_count" in feats:
        h = hist("row_count")
        if h["z"] is not None and h["z"] < -THR:
            return "missing_partial_day", calls
    for f in feats:
        if f.endswith(".null_rate"):
            h = hist(f)
            if h["today"] is not None and h["baseline_median"] is not None and h["today"] - h["baseline_median"] > 0.02:
                return "null_spike", calls
    for col in ("trip_distance", "fare_amount"):
        if any(f.startswith(col + ".") for f in feats):
            ratios = []
            for stat in ("mean", "median", "p95"):
                h = hist(f"{col}.{stat}")
                if h["ratio"] is not None:
                    ratios.append(h["ratio"])
            if len(ratios) == 3 and (max(ratios) - min(ratios)) < 0.12 and abs(sum(ratios) / 3 - 1) > 0.15:
                return "unit_change", calls
    return "distribution_drift", calls
