## Detection (48 faulty days, 48 clean days; 2025-12-01 to 2026-01-31)

| Detector | Detection rate | False alarms, all clean days | False alarms, ordinary days (43) |
|---|---|---|---|
| Fixed threshold | 75% (36/48) | 33% (16/48) | 26% (11/43) |
| Rolling z-score | 98% (47/48) | 25% (12/48) | 23% (10/43) |
| Median/MAD + KS/PSI (ours) | 96% (46/48) | 25% (12/48) | 21% (9/43) |

Documented events excluded from 'ordinary': 2025-12-25 (Christmas Day), 2026-01-01 (New Year's Day), 2026-01-19 (Martin Luther King Jr. Day), 2026-01-25 (NYC winter storm), 2026-01-26 (NYC winter storm)

Clean days flagged by each detector:
- Fixed threshold: 2025-12-25, 2025-12-26, 2025-12-27, 2025-12-28, 2025-12-31, 2026-01-01, 2026-01-10, 2026-01-11, 2026-01-17, 2026-01-18, 2026-01-19, 2026-01-24, 2026-01-25, 2026-01-26, 2026-01-27, 2026-01-31
- Rolling z-score: 2025-12-20, 2025-12-22, 2025-12-24, 2025-12-25, 2025-12-28, 2025-12-29, 2026-01-04, 2026-01-05, 2026-01-10, 2026-01-12, 2026-01-25, 2026-01-27
- Median/MAD + KS/PSI (ours): 2025-12-20, 2025-12-23, 2025-12-24, 2025-12-25, 2025-12-28, 2026-01-10, 2026-01-25, 2026-01-26, 2026-01-27, 2026-01-28, 2026-01-29, 2026-01-30

## Root-cause accuracy (on days our detector flagged)

| Setup | Cause accuracy | End-to-end (all 48 faulty) | Avg tool calls |
|---|---|---|---|
| Scripted baseline: alert text only | 67% (31/46) | 65% (31/48) | 0.0 |
| Scripted baseline: metric history | 56% (26/46) | 54% (26/48) | 1.41 |
| Full tool-using agent | 72% (33/46) | 69% (33/48) | 3.67 |

## Accuracy per fault type (cause accuracy)

| Fault | Scripted baseline: alert text only | Scripted baseline: metric history | Full tool-using agent |
|---|---|---|---|
| null_spike | 100% (8/8) | 88% (7/8) | 88% (7/8) |
| duplicate_rows | 100% (8/8) | 100% (8/8) | 100% (8/8) |
| schema_change | 100% (8/8) | 0% (0/8) | 100% (8/8) |
| distribution_drift | 100% (7/7) | 71% (5/7) | 71% (5/7) |
| missing_partial_day | 0% (0/7) | 57% (4/7) | 57% (4/7) |
| unit_change | 0% (0/8) | 25% (2/8) | 12% (1/8) |

## Detection rate per fault type

| Fault | Fixed threshold | Rolling z-score | Median/MAD + KS/PSI (ours) |
|---|---|---|---|
| null_spike | 88% (7/8) | 100% (8/8) | 100% (8/8) |
| duplicate_rows | 100% (8/8) | 100% (8/8) | 100% (8/8) |
| schema_change | 100% (8/8) | 100% (8/8) | 100% (8/8) |
| distribution_drift | 0% (0/8) | 88% (7/8) | 88% (7/8) |
| missing_partial_day | 62% (5/8) | 100% (8/8) | 88% (7/8) |
| unit_change | 100% (8/8) | 100% (8/8) | 100% (8/8) |
