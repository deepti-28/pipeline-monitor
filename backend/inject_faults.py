"""The 6 fault types. inject(df, fault, rng) -> (faulty_df, info).

Severity is randomised so some faults are subtle: that is what makes the detector
comparison meaningful.
"""
import numpy as np
import pandas as pd

from .config import FAULTS

MILES_TO_KM = 1.609


def inject(df: pd.DataFrame, fault: str, rng: np.random.Generator):
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")
    d = df.copy()
    n = len(d)
    info = {"fault": fault}

    if fault == "null_spike":
        col = str(rng.choice(["fare_amount", "trip_distance", "payment_type", "PULocationID"]))
        frac = float(rng.uniform(0.04, 0.40))
        d[col] = d[col].astype(float)
        d.loc[rng.random(n) < frac, col] = np.nan
        info.update(column=col, fraction=round(frac, 3))

    elif fault == "duplicate_rows":
        frac = float(rng.uniform(0.03, 0.25))
        dup = d.sample(frac=frac, random_state=int(rng.integers(1e9)))
        d = pd.concat([d, dup]).sort_values("tpep_pickup_datetime", kind="stable").reset_index(drop=True)
        info.update(fraction=round(frac, 3))

    elif fault == "schema_change":
        col = str(rng.choice(["trip_distance", "fare_amount", "payment_type"]))
        if rng.random() < 0.5:
            d = d.rename(columns={col: col + "_v2"})
            info.update(column=col, change="renamed")
        else:
            d = d.drop(columns=[col])
            info.update(column=col, change="dropped")

    elif fault == "distribution_drift":
        choice = str(rng.choice(["payment_type", "PULocationID", "passenger_count", "fare_amount"]))
        frac = float(rng.uniform(0.20, 0.60))
        mask = rng.random(n) < frac
        if choice == "payment_type":
            d.loc[mask, "payment_type"] = 2
        elif choice == "PULocationID":
            d.loc[mask, "PULocationID"] = rng.integers(1, 6, mask.sum())
        elif choice == "passenger_count":
            d.loc[mask, "passenger_count"] = 4.0
        else:
            d.loc[mask, "fare_amount"] = d.loc[mask, "fare_amount"] + 8.0
        info.update(column=choice, fraction=round(frac, 3))

    elif fault == "missing_partial_day":
        keep = float(rng.uniform(0.30, 0.85))
        d = d.iloc[: max(50, int(n * keep))].reset_index(drop=True)
        info.update(kept_fraction=round(keep, 3))

    elif fault == "unit_change":
        d["trip_distance"] = np.round(d["trip_distance"] * MILES_TO_KM, 2)
        info.update(column="trip_distance", factor=MILES_TO_KM)

    return d, info
