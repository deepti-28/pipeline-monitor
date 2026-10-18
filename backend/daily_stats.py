import pandas as pd
from backend.evaluate import load_real_days

dates, dfs, snaps = load_real_days()
rows = []
for d, s in zip(dates, snaps):
    c = s["columns"]
    rows.append({
        "date": d, "day": pd.Timestamp(d).day_name()[:3], "rows": s["row_count"],
        "fare_mean": c["fare_amount"].get("mean"), "fare_median": c["fare_amount"].get("median"),
        "dist_mean": c["trip_distance"].get("mean"),
        "pax_null": c["passenger_count"].get("null_rate"),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))