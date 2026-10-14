"""Per-day data-quality snapshot (the 'metrics computer')."""
import numpy as np
import pandas as pd

from .config import CAT_COLS, EXPECTED_COLS, MONITOR_KEYS, NUM_STAT_COLS


def _clean(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(x) or np.isinf(x)) else x


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with every expected column present (missing ones filled with NaN)."""
    out = df.copy()
    for c in EXPECTED_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out


def compute_snapshot(df: pd.DataFrame, date_str: str) -> dict:
    raw_cols = sorted(df.columns)
    d = conform(df)
    n = len(d)
    snap = {
        "date": date_str,
        "dow": pd.Timestamp(date_str).weekday(),
        "row_count": int(n),
        "dup_rate": _clean(d.duplicated().mean()) if n else 0.0,
        "schema": raw_cols,
        "columns": {},
    }
    for c in EXPECTED_COLS:
        s = d[c]
        m = {"null_rate": _clean(s.isna().mean()) if n else None}
        if c in NUM_STAT_COLS:
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                m.update(
                    mean=_clean(v.mean()),
                    std=_clean(v.std()),
                    min=_clean(v.min()),
                    max=_clean(v.max()),
                    median=_clean(v.median()),
                    p95=_clean(v.quantile(0.95)),
                )
            else:
                m.update(mean=None, std=None, min=None, max=None, median=None, p95=None)
        if c in CAT_COLS:
            vc = pd.to_numeric(s, errors="coerce").dropna().value_counts(normalize=True)
            m["distinct"] = int(len(vc))
            m["top_share"] = _clean(vc.iloc[0]) if len(vc) else None
            m["top_value"] = str(vc.index[0]) if len(vc) else None
        snap["columns"][c] = m
    return snap


def flatten(snap: dict) -> dict:
    """Monitored scalar features: {'row_count': .., 'fare_amount.mean': ..}."""
    f = {"row_count": float(snap["row_count"])}
    if snap.get("dup_rate") is not None:
        f["dup_rate"] = float(snap["dup_rate"])
    for c, keys in MONITOR_KEYS.items():
        m = snap["columns"].get(c, {})
        for k in keys:
            if m.get(k) is not None:
                f[f"{c}.{k}"] = float(m[k])
    return f
