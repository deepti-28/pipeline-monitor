"""Create data/daily/*: one file per day, from NYC yellow-taxi parquet files in data/raw/.

Falls back to synthetic data (backend/synth.py) if no raw files exist,
so a fresh deploy (e.g. on Render) still starts with data.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DAILY = ROOT / "data" / "daily"

TIME_COL = "tpep_pickup_datetime"
COLUMNS = [TIME_COL, "passenger_count", "trip_distance", "fare_amount", "payment_type", "PULocationID"]

# Cleaning (set CLEAN = False to keep everything)
CLEAN = True
MAX_DISTANCE = 100    # miles
MAX_FARE = 1000       # dollars

# Fraction of each day's trips to keep. 1.0 keeps everything;
# the old version used 0.04 (~100k trips/day -> ~4k).
SAMPLE_FRAC = 1.0
MIN_ROWS = 1000       # skip days too small to compute stable metrics


def daily_files():
    return sorted(list(DAILY.glob("*.parquet")) + list(DAILY.glob("*.csv")))


def read_day(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=[TIME_COL])


def _write(df: pd.DataFrame, day: str):
    DAILY.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        df.to_parquet(DAILY / f"{day}.parquet", index=False)
    else:
        df.to_csv(DAILY / f"{day}.csv", index=False)


def split_month(path):
    """Split one raw parquet file into one file per pickup day."""
    df = pd.read_parquet(path, columns=COLUMNS)
    df = df.dropna(subset=[TIME_COL])
    if CLEAN:
        df = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= MAX_DISTANCE)]
        df = df[(df["fare_amount"] >= 0) & (df["fare_amount"] <= MAX_FARE)]
    df = df.assign(day=df[TIME_COL].dt.date.astype(str))
    # keep only the modal month (real files contain a few stray dates)
    month = df["day"].str[:7].mode()[0]
    df = df[df["day"].str.startswith(month)]

    written = 0
    for day, g in df.groupby("day"):
        if SAMPLE_FRAC < 1.0:
            g = g.sample(frac=SAMPLE_FRAC, random_state=0)
        if len(g) < MIN_ROWS:
            continue
        g = g.drop(columns="day").sort_values(TIME_COL).reset_index(drop=True)
        _write(g, day)
        written += 1
    print(f"split {path} -> {written} daily files (month {month})", file=sys.stderr)


def make_synthetic(n_days: int = 45, seed: int = 42, start: str = "2024-03-01"):
    import numpy as np
    from .synth import make_day  # only needed if there is no raw data
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
        for p in raw:
            split_month(p)
    else:
        make_synthetic()


def main():
    ensure_daily_files()
    print(f"{len(daily_files())} daily files in {DAILY}", file=sys.stderr)


if __name__ == "__main__":
    main()