import collections
import pandas as pd
from backend.evaluate import load_real_days, DETECTORS
from backend.config import WINDOW

dates, dfs, snaps = load_real_days()
det = DETECTORS["Median/MAD + KS/PSI (ours)"]
counter = collections.Counter()

for w in range(len(dfs) - WINDOW):
    hist = snaps[w:w + WINDOW]
    ref = pd.concat([x.sample(min(400, len(x)), random_state=w) for x in dfs[w:w + WINDOW]])
    day = dates[w + WINDOW]
    a = det(hist, snaps[w + WINDOW], today_df=dfs[w + WINDOW], ref_df=ref)
    if a["is_anomaly"]:
        top = sorted(a["flags"], key=lambda f: -abs(f["score"]))[:4]
        print(day, pd.Timestamp(day).day_name(), [(f["feature"], round(f["score"], 1)) for f in top])
        counter.update(f["feature"] for f in a["flags"])

print("\nMost frequent false-alarm features:", counter.most_common(10))