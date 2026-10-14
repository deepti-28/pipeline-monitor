"""Synthetic NYC-yellow-taxi-shaped daily data (same columns as the real parquet).

Used when no real file is present in data/raw/. Weekends carry ~22% less volume so the
day-of-week adjustment in the detector has something real to correct for.
"""
import numpy as np
import pandas as pd

BASE_ROWS = 4000


def _zone_probs() -> np.ndarray:
    p = 1.0 / np.arange(1, 61) ** 0.8
    p = p / p.sum()
    return np.random.default_rng(0).permutation(p)


ZONE_P = _zone_probs()


def weekday_factor(d) -> float:
    return 0.78 if pd.Timestamp(d).weekday() >= 5 else 1.0


def make_day(d, rng: np.random.Generator, base: int = BASE_ROWS) -> pd.DataFrame:
    n = max(500, int(base * weekday_factor(d) * rng.normal(1.0, 0.03)))
    secs = np.sort(rng.integers(0, 86400, n))
    ts = pd.Timestamp(d) + pd.to_timedelta(secs, unit="s")
    dist = np.round(np.clip(rng.lognormal(0.75, 0.8, n), 0.1, 60), 2)
    fare = np.round(np.clip(3 + 2.7 * dist + rng.normal(0, 1.5, n), 3, None), 2)
    pc = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.70, 0.15, 0.06, 0.02, 0.07]).astype(float)
    pc[rng.random(n) < 0.02] = np.nan
    pay = rng.choice([1, 2, 3, 4], size=n, p=[0.72, 0.25, 0.02, 0.01])
    zone = rng.choice(np.arange(1, 61), size=n, p=ZONE_P)
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": ts,
            "passenger_count": pc,
            "trip_distance": dist,
            "fare_amount": fare,
            "payment_type": pay,
            "PULocationID": zone,
        }
    )
