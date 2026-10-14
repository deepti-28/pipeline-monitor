"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DAILY = DATA / "daily"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
DB_PATH = DATA / "monitor.db"

WINDOW = 14  # days of clean history used as the baseline

EXPECTED_COLS = [
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "payment_type",
    "PULocationID",
]
NUM_STAT_COLS = ["passenger_count", "trip_distance", "fare_amount"]  # mean/std/min/max/median/p95
CONT_COLS = ["trip_distance", "fare_amount"]  # continuous: KS test, quantile ratios
CAT_COLS = ["passenger_count", "payment_type", "PULocationID"]  # distinct / top share / PSI
KEY_COLS = ["tpep_pickup_datetime", "PULocationID", "fare_amount", "trip_distance"]

# Features the detectors watch per column (min/max are stored but too noisy to alert on).
MONITOR_KEYS = {
    "tpep_pickup_datetime": ["null_rate"],
    "passenger_count": ["null_rate", "mean", "distinct", "top_share"],
    "trip_distance": ["null_rate", "mean", "std", "median", "p95"],
    "fare_amount": ["null_rate", "mean", "std", "median", "p95"],
    "payment_type": ["null_rate", "distinct", "top_share"],
    "PULocationID": ["null_rate", "distinct", "top_share"],
}

FAULTS = [
    "null_spike",
    "duplicate_rows",
    "schema_change",
    "distribution_drift",
    "missing_partial_day",
    "unit_change",
]
