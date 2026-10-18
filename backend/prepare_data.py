"""Split raw NYC TLC Yellow Taxi monthly files into one parquet file per day.

Usage:  python -m backend.prepare_data

Reads every data/raw/yellow_tripdata_YYYY-MM.parquet, keeps only trips whose pickup time
falls inside that month (TLC files contain stray dates), optionally drops physically
impossible rows, and writes data/daily/YYYY-MM-DD.parquet. Existing daily files are
replaced, so the folder always matches the raw files.
"""
import re

import pandas as pd

from .config import ROOT

RAW = ROOT / "data" / "raw"
DAILY = ROOT / "data" / "daily"

TIME_COL = "tpep_pickup_datetime"
COLUMNS = [TIME_COL, "passenger_count", "trip_distance", "fare_amount", "payment_type", "PULocationID"]

# Light cleaning of physically impossible values (negative fares, trips of hundreds of miles).
# Nulls are NOT removed: natural nulls (e.g. passenger_count) are part of the real data.
# Describe this step in the README. Set CLEAN = False to keep every row.
CLEAN = True
MAX_DISTANCE = 100    # miles
MAX_FARE = 1000       # dollars


def split_month(path):
    m = re.search(r"(\d{4})-(\d{2})", path.stem)
    if not m:
        print(f"skip {path.name}: no YYYY-MM in the file name")
        return 0
    start = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    end = start + pd.offsets.MonthBegin(1)

    df = pd.read_parquet(path, columns=COLUMNS)
    n_raw = len(df)
    df = df[(df[TIME_COL] >= start) & (df[TIME_COL] < end)]
    n_month = len(df)
    if CLEAN:
        df = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= MAX_DISTANCE)
                & (df["fare_amount"] >= 0) & (df["fare_amount"] <= MAX_FARE)]
    print(f"{path.name}: {n_raw:,} rows -> {n_month:,} in month -> {len(df):,} after cleaning")

    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64"):
        df[c] = df[c].astype("int32")

    days = df[TIME_COL].dt.strftime("%Y-%m-%d")
    count = 0
    for day, g in df.groupby(days):
        g.to_parquet(DAILY / f"{day}.parquet", index=False)
        count += 1
    return count


def main():
    files = sorted(RAW.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise SystemExit(f"No raw files found in {RAW}")
    DAILY.mkdir(parents=True, exist_ok=True)
    for old in DAILY.glob("*.parquet"):
        old.unlink()
    total = sum(split_month(f) for f in files)
    print(f"Wrote {total} daily files to {DAILY}")


if __name__ == "__main__":
    main()