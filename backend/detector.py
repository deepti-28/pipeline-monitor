"""Three detectors with one interface.

Each returns {"detector", "date", "is_anomaly", "flags": [...], "text"}; a flag is
{"feature", "kind", "value", "baseline", "score"}. `hist` is a list of clean snapshots,
`today` is today's snapshot.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import CAT_COLS, CONT_COLS
from .metrics import flatten

THR = 5.0  # robust-z threshold
KS_THR = 0.08
PSI_THR = 0.2

# Relative noise floors: sampling noise differs by statistic (std/p95 are noisier than mean).
REL_FLOOR = {"mean": 0.02, "median": 0.02, "p95": 0.03, "std": 0.06, "distinct": 0.02,
             "top_share": 0.02, "row_count": 0.03}
RATE_SUFFIXES = ("null_rate", "dup_rate", "top_share")


def _mad(a) -> float:
    a = np.asarray(a, float)
    return float(np.median(np.abs(a - np.median(a))))


def robust_stats(vals, feat):
    vals = np.asarray(vals, float)
    med = float(np.median(vals))
    suffix = feat.split(".")[-1]
    rel = REL_FLOOR.get(suffix, 0.02)
    abs_floor = 0.005 if suffix in RATE_SUFFIXES else 1e-6
    scale = max(1.4826 * _mad(vals), rel * abs(med), abs_floor)
    return med, scale


def baseline_from(hist, hflat, feat, weekend: bool):
    """(median, scale) for a feature. row_count is adjusted for weekday vs weekend."""
    series = [h[feat] for h in hflat if feat in h]
    if feat == "row_count":
        groups = {True: [], False: []}
        for snap, h in zip(hist, hflat):
            groups[snap["dow"] >= 5].append(h["row_count"])
        if len(groups[weekend]) >= 3:
            med_g = {k: float(np.median(v)) for k, v in groups.items() if v}
            resid = [x - med_g[k] for k, v in groups.items() for x in v]
            med = med_g[weekend]
            scale = max(1.4826 * _mad(resid), REL_FLOOR["row_count"] * med, 1.0)
            return med, scale
    if len(series) < 5:
        return None, None
    return robust_stats(series, feat)


def score_feature(hist, today, feat):
    """Used by the agent's get_metric_history tool. Returns series, baseline, z."""
    hflat = [flatten(h) for h in hist]
    tf = flatten(today)
    med, scale = baseline_from(hist, hflat, feat, today["dow"] >= 5)
    x = tf.get(feat)
    z = None if (x is None or med is None) else (x - med) / scale
    return {"series": [h[feat] for h in hflat if feat in h], "baseline": med, "scale": scale,
            "today": x, "z": z}


def _fmt(v):
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer() and abs(v) > 50):
        return f"{int(v)}"
    return f"{v:.4g}"


def _alert(name, today, flags):
    flags = sorted(flags, key=lambda f: -abs(f["score"]))
    parts = []
    for f in flags[:4]:
        if f["kind"] == "schema":
            parts.append(f"schema changed (removed {f['detail']['removed']}, added {f['detail']['added']})")
        else:
            b = "n/a" if f["baseline"] is None else _fmt(f["baseline"])
            parts.append(f"{f['feature']}={_fmt(f['value'])} (baseline {b}, score {f['score']:.1f})")
    text = f"[{name}] {today['date']}: " + ("; ".join(parts) if parts else "no anomaly")
    return {"detector": name, "date": today["date"], "is_anomaly": bool(flags), "flags": flags, "text": text}


# ---------------------------------------------------------------- Baseline A
def detect_fixed(hist, today, **_):
    flags = []
    for c, m in today["columns"].items():
        nr = m.get("null_rate")
        if nr is not None and nr > 0.05:
            flags.append(dict(feature=f"{c}.null_rate", kind="rule", value=nr, baseline=0.05, score=nr / 0.05))
    if (today.get("dup_rate") or 0) > 0.01:
        flags.append(dict(feature="dup_rate", kind="rule", value=today["dup_rate"], baseline=0.01,
                          score=today["dup_rate"] / 0.01))
    rc = today["row_count"]
    if rc < 2000 or rc > 7000:
        flags.append(dict(feature="row_count", kind="rule", value=rc, baseline=4000, score=(rc - 4000) / 2000))
    for col, lo, hi in (("fare_amount", 6, 20), ("trip_distance", 1, 6)):
        mean = today["columns"][col].get("mean")
        if mean is not None and not (lo <= mean <= hi):
            flags.append(dict(feature=f"{col}.mean", kind="rule", value=mean, baseline=(lo + hi) / 2,
                              score=abs(mean - (lo + hi) / 2) / ((hi - lo) / 2)))
    return _alert("Fixed threshold", today, flags)


# ---------------------------------------------------------------- Baseline B
def detect_rolling_z(hist, today, thr: float = 3.0, **_):
    hflat = [flatten(h) for h in hist]
    tf = flatten(today)
    flags = []
    for feat, x in tf.items():
        s = [h[feat] for h in hflat if feat in h]
        if len(s) < 5:
            continue
        mu, sd = float(np.mean(s)), float(np.std(s, ddof=1))
        z = (x - mu) / max(sd, 1e-9)
        if abs(z) > thr:
            flags.append(dict(feature=feat, kind="z", value=x, baseline=mu, score=z))
    return _alert("Rolling z-score", today, flags)


# ---------------------------------------------------------------- Ours
def _sample(series, n, rng):
    a = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    return a if len(a) <= n else rng.choice(a, n, replace=False)


def psi(a_series, b_series):
    a = pd.to_numeric(a_series, errors="coerce").dropna()
    b = pd.to_numeric(b_series, errors="coerce").dropna()
    if len(a) < 200 or len(b) < 200:
        return None
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    cats = pa.index.union(pb.index)
    pa = pa.reindex(cats).fillna(0).to_numpy() + 1e-4
    pb = pb.reindex(cats).fillna(0).to_numpy() + 1e-4
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def detect_robust(hist, today, today_df=None, ref_df=None, thr: float = THR, **_):
    flags = []
    if hist:
        prev, cur = set(hist[-1]["schema"]), set(today["schema"])
        if prev != cur:
            flags.append(dict(feature="schema", kind="schema", value=len(prev ^ cur), baseline=0, score=999.0,
                              detail={"removed": sorted(prev - cur), "added": sorted(cur - prev)}))
    hflat = [flatten(h) for h in hist]
    tf = flatten(today)
    weekend = today["dow"] >= 5
    for feat, x in tf.items():
        med, scale = baseline_from(hist, hflat, feat, weekend)
        if med is None:
            continue
        z = (x - med) / scale
        if abs(z) > thr:
            flags.append(dict(feature=feat, kind="z", value=x, baseline=med, score=z))
    if today_df is not None and ref_df is not None:
        rng = np.random.default_rng(0)
        for col in CONT_COLS:
            if col in today_df.columns and col in ref_df.columns:
                a, b = _sample(today_df[col], 3000, rng), _sample(ref_df[col], 5000, rng)
                if len(a) >= 200 and len(b) >= 200:
                    ks = float(ks_2samp(a, b).statistic)
                    if ks > KS_THR:
                        flags.append(dict(feature=f"{col}.ks", kind="ks", value=ks, baseline=0.0,
                                          score=ks / KS_THR * thr))
        for col in CAT_COLS:
            if col in today_df.columns and col in ref_df.columns:
                p = psi(today_df[col], ref_df[col])
                if p is not None and p > PSI_THR:
                    flags.append(dict(feature=f"{col}.psi", kind="psi", value=p, baseline=0.0,
                                      score=p / PSI_THR * thr))
    return _alert("Median/MAD + KS/PSI", today, flags)
