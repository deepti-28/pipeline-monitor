## Detection (30 faulty days, 30 clean days)

| Detector | Detection rate | False alarm rate |
|---|---|---|
| Fixed threshold | 63% (19/30) | 0% (0/30) |
| Rolling z-score | 97% (29/30) | 7% (2/30) |
| Median/MAD + KS/PSI (ours) | 100% (30/30) | 0% (0/30) |

## Root-cause accuracy (on days our detector flagged)

| Setup | Cause accuracy | End-to-end (all 30 faulty) | Avg tool calls |
|---|---|---|---|
| LLM-style: alert text only | 83% (25/30) | 83% (25/30) | 0.0 |
| LLM-style: metric history | 83% (25/30) | 83% (25/30) | 1.17 |
| Full tool-using agent | 100% (30/30) | 100% (30/30) | 3.5 |

## Accuracy per fault type (cause accuracy)

| Fault | LLM-style: alert text only | LLM-style: metric history | Full tool-using agent |
|---|---|---|---|
| null_spike | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| duplicate_rows | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| schema_change | 100% (5/5) | 0% (0/5) | 100% (5/5) |
| distribution_drift | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| missing_partial_day | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| unit_change | 0% (0/5) | 100% (5/5) | 100% (5/5) |

## Detection rate per fault type

| Fault | Fixed threshold | Rolling z-score | Median/MAD + KS/PSI (ours) |
|---|---|---|---|
| null_spike | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| duplicate_rows | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| schema_change | 100% (5/5) | 100% (5/5) | 100% (5/5) |
| distribution_drift | 0% (0/5) | 100% (5/5) | 100% (5/5) |
| missing_partial_day | 80% (4/5) | 80% (4/5) | 100% (5/5) |
| unit_change | 0% (0/5) | 100% (5/5) | 100% (5/5) |
