"""The six investigation tools. Each takes a Ctx plus keyword args and returns a JSON-safe dict."""
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..config import CAT_COLS, CONT_COLS, KEY_COLS
from ..detector import THR, baseline_from, psi, score_feature
from ..metrics import flatten


@dataclass
class Ctx:
    hist: list  # clean baseline snapshots (oldest -> newest)
    today: dict  # today's snapshot
    today_df: Optional[pd.DataFrame] = None
    ref_df: Optional[pd.DataFrame] = None  # sample of recent clean rows
    hist_all: list = field(default_factory=list)  # recent snapshots incl. flagged days


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def get_metric_history(ctx: Ctx, feature: str, n: int = 14):
    s = score_feature(ctx.hist, ctx.today, feature)
    base = s["baseline"]
    return {
        "feature": feature,
        "history": [_r(v) for v in s["series"][-n:]],
        "baseline_median": _r(base),
        "today": _r(s["today"]),
        "z": _r(s["z"], 2),
        "ratio": _r(s["today"] / base, 3) if (s["today"] is not None and base) else None,
    }


def compare_schema(ctx: Ctx):
    prev = set(ctx.hist[-1]["schema"]) if ctx.hist else set()
    cur = set(ctx.today["schema"])
    removed, added = sorted(prev - cur), sorted(cur - prev)
    return {"changed": bool(removed or added), "removed": removed, "added": added,
            "possible_rename": bool(removed and added)}


def find_first_bad_day(ctx: Ctx, feature: str):
    days = list(ctx.hist_all or ctx.hist) + [ctx.today]
    hflat = [flatten(h) for h in ctx.hist]
    run_start, run_len = None, 0
    for snap in reversed(days):
        x = flatten(snap).get(feature)
        med, scale = baseline_from(ctx.hist, hflat, feature, snap["dow"] >= 5)
        bad = x is None or (med is not None and abs((x - med) / scale) > THR)
        if not bad:
            break
        run_start, run_len = snap["date"], run_len + 1
    return {"feature": feature, "first_bad_day": run_start, "consecutive_bad_days": run_len}


def _key_cols(df):
    return [k for k in KEY_COLS if k in df.columns and df[k].notna().any()]


def count_duplicate_keys(ctx: Ctx):
    df = ctx.today_df
    keys = _key_cols(df)
    dup = df.duplicated(subset=keys, keep="first")
    hist_rates = [h["dup_rate"] for h in ctx.hist if h.get("dup_rate") is not None]
    return {"key_columns": keys, "duplicate_rows": int(dup.sum()), "dup_rate": _r(dup.mean()),
            "history_median_dup_rate": _r(np.median(hist_rates)) if hist_rates else None}


def sample_bad_rows(ctx: Ctx, kind: str, column: Optional[str] = None, k: int = 5):
    df = ctx.today_df
    if kind == "duplicate":
        bad = df[df.duplicated(subset=_key_cols(df), keep=False)]
    elif kind == "null" and column in df.columns:
        bad = df[df[column].isna()]
    else:
        return {"kind": kind, "column": column, "count": 0, "rows": []}
    rows = json.loads(bad.head(k).to_json(orient="records", date_format="iso"))
    return {"kind": kind, "column": column, "count": int(len(bad)), "rows": rows}


def compare_distribution(ctx: Ctx, column: str):
    if ctx.today_df is None or ctx.ref_df is None or column not in ctx.today_df.columns:
        return {"column": column, "error": "column missing or no reference data"}
    t = pd.to_numeric(ctx.today_df[column], errors="coerce").dropna()
    r = pd.to_numeric(ctx.ref_df[column], errors="coerce").dropna()
    if len(t) < 100 or len(r) < 100:
        return {"column": column, "error": "not enough data"}
    if column in CONT_COLS:
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        ratios = [float(t.quantile(q) / r.quantile(q)) for q in qs if r.quantile(q) > 0]
        rng = np.random.default_rng(0)
        a = t.to_numpy() if len(t) <= 3000 else rng.choice(t.to_numpy(), 3000, replace=False)
        b = r.to_numpy() if len(r) <= 5000 else rng.choice(r.to_numpy(), 5000, replace=False)
        return {
            "column": column,
            "ks": _r(ks_2samp(a, b).statistic),
            "quantile_ratios": [_r(x, 3) for x in ratios],
            "median_ratio": _r(t.median() / r.median(), 3),
            # near-zero spread + ratio far from 1 = the whole distribution was scaled
            "quantile_ratio_cv": _r(np.std(ratios) / np.mean(ratios), 4),
        }
    value = psi(t, r)
    pt, pr = t.value_counts(normalize=True), r.value_counts(normalize=True)
    cats = pt.index.union(pr.index)
    shift = (pt.reindex(cats).fillna(0) - pr.reindex(cats).fillna(0)).sort_values(key=abs, ascending=False)
    top = [{"category": str(c), "today": _r(pt.get(c, 0)), "reference": _r(pr.get(c, 0))} for c in shift.index[:3]]
    return {"column": column, "psi": _r(value), "largest_shifts": top}


TOOLS = {
    "get_metric_history": get_metric_history,
    "compare_schema": compare_schema,
    "find_first_bad_day": find_first_bad_day,
    "count_duplicate_keys": count_duplicate_keys,
    "sample_bad_rows": sample_bad_rows,
    "compare_distribution": compare_distribution,
}
