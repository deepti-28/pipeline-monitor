"""Create data/daily/*: one file per day.

- If a real NYC yellow-taxi parquet is in data/raw/, split it by pickup day and keep a
  fixed SAMPLE_FRAC of each day's trips (a real day has ~100k trips; the app is tuned
  for a few thousand, and a fixed fraction keeps weekday/weekend volume patterns).
- Otherwise generate synthetic taxi-shaped data (45 days).
Files are parquet when pyarrow is installed, CSV otherwise; readers handle both.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import DAILY, EXPECTED_COLS, RAW
from .synth import make_day

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Fraction of each real day's trips to keep. 0.04 turns ~100k trips/day into ~4k.
# Use 1.0 to keep everything (slow), or lower it if your file has more trips per day.
SAMPLE_FRAC = 0.04


def daily_files():
    return sorted(list(DAILY.glob("*.parquet")) + list(DAILY.glob("*.csv")))


def read_day(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["tpep_pickup_datetime"])
    return df


def _write(df: pd.DataFrame, day: str):
    DAILY.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        df.to_parquet(DAILY / f"{day}.parquet", index=False)
    else:
        df.to_csv(DAILY / f"{day}.csv", index=False)


def split_real(path, min_rows: int = 1000, frac: float = SAMPLE_FRAC):
    df = pd.read_parquet(path, columns=EXPECTED_COLS)
    df = df.dropna(subset=["tpep_pickup_datetime"])
    df["day"] = df["tpep_pickup_datetime"].dt.date.astype(str)
    # keep only the modal month (real files contain a few stray dates)
    month = df["day"].str[:7].mode()[0]
    df = df[df["day"].str.startswith(month)]
    written = 0
    for day, g in df.groupby("day"):
        g = g.sample(frac=frac, random_state=0) if frac < 1.0 else g
        if len(g) >= min_rows:  # skip days that are too small to compute stable metrics
            g = g.drop(columns="day").sort_values("tpep_pickup_datetime").reset_index(drop=True)
            _write(g, day)
            written += 1
    print(f"split {path} -> {written} daily files (month {month}, sample {frac:.0%})", file=sys.stderr)


def make_synthetic(n_days: int = 45, seed: int = 42, start: str = "2024-03-01"):
    rng = np.random.default_rng(seed)
    d0 = date.fromisoformat(start)
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        _write(make_day(d, rng), str(d))


def ensure_daily_files():
    if daily_files():
        return
    raw = sorted(RAW.glob("*.parquet")) if RAW.exists() else []
    if raw:
        split_real(raw[0])
    else:
        make_synthetic()


if __name__ == "__main__":
    ensure_daily_files()
    print(f"{len(daily_files())} daily files in {DAILY}", file=sys.stderr)