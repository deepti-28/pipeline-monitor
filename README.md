# Pipeline Health Monitor — with a Root-Cause Agent

A web app that loads a day of data at a time, computes quality metrics, detects data-quality
failures, investigates each one with tools to find the root cause, and **measures its own
accuracy on injected faults**. Click *Run next day* / *Queue fault* in the dashboard to watch it work.

## Architecture

```
Daily file -> Loader -> SQLite (trips table)
                 |
         Metrics computer -> daily_metrics
                 |
          Detector (median/MAD + KS/PSI) -> alert?
                 |
          Agent (6 tools, fixed JSON output) -> cause, evidence, fix
                 |
      FastAPI -> alerts / agent_runs -> WebSocket -> dashboard (served by the same app)
```

## Run it

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload     # open http://127.0.0.1:8000
python -m backend.evaluate            # re-run all experiments -> results/
python -m pytest tests                # tests
```

On first start the app creates `data/daily/` (45 daily files), loads the first 14 days as the
warm-up baseline, and shows the pre-computed evaluation. **To use the real NYC Yellow Taxi data**, put one
month's parquet file in `data/raw/`, delete `data/daily/` and `data/monitor.db`, and restart; it is split into
daily files automatically (same six columns).

## Fault types

`null_spike`, `duplicate_rows`, `schema_change` (renamed/dropped column), `distribution_drift`,
`missing_partial_day`, `unit_change` (miles to km). Severity is randomised per case so some faults are subtle.

## Detectors

1. **Fixed threshold**: null rate > 5%, duplicate rate > 1%, row count outside [2000, 7000], mean fare/distance outside fixed bounds.
2. **Rolling z-score**: |z| > 3 against the last 14 days.
3. **Ours**: median/MAD robust z with per-statistic noise floors, a weekday/weekend adjustment for row count,
   schema diff, plus KS (continuous columns) and PSI (categorical columns) against recent clean rows.
   Baselines exclude days that already alerted.

## Agent

Six tools (`get_metric_history`, `compare_schema`, `find_first_bad_day`, `count_duplicate_keys`,
`sample_bad_rows`, `compare_distribution`) and a fixed output `{cause, evidence, confidence, suggested_fix}`
with `cause` from the six categories. Every tool call is stored and shown on the Alert detail page.

## Results

Produced by `python -m backend.evaluate` (30 faulty days, 5 per type, plus 30 clean days; each case = 14 clean
history days + 1 test day):

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

## Honest limitations (read before quoting numbers)

- **Synthetic data by default.** The numbers above come from taxi-shaped synthetic data, not the real NYC file.
  Real data is messier (natural nulls, outliers, holidays), so expect lower scores. Re-run on the real file before using any number on a resume.
- **Self-evaluation.** I tuned the detector thresholds while looking at the same fault generator it is tested on.
  100% detection is therefore optimistic; the fault generator is the main thing to make harder
  (smaller shifts, combined faults, faults on holidays).
- **The agent is not an LLM.** The planner is a deterministic investigation policy over the six tools, and the two
  "LLM-style" baselines are scripted analyzers that see less information. They are stand-ins, not LLM calls.
  The tools and JSON schema are LLM-ready: replace `run_agent` in `backend/agent/agent.py` with a
  LangChain/Anthropic tool-calling loop and re-run `evaluate.py` to get the real LLM comparison.
- **Small sample.** 5 cases per fault type; 95% Wilson intervals are in `results/eval_results.json` and are wide.
- **MongoDB is replaced by a SQLite document store** (`backend/db.py`) so the app needs no external service;
  the interface is Mongo-shaped for a later swap to Atlas.
- The HTTP/WebSocket layer and dashboard were written in an environment where `fastapi` could not be installed,
  so run `python -m pytest tests/test_api.py` once on your machine.

## Deploy

`render.yaml` deploys the whole app (API + dashboard) as one Render web service. The free tier sleeps when idle and the
SQLite file resets on restart; the dashboard shows a "backend waking up" message and the app re-creates its demo state on boot.
