"""Tool-using root-cause agent.

The planner here is a deterministic investigation policy: it decides which tool to call
next from the previous tool result, records every call, and emits the fixed JSON
{cause, evidence, confidence, suggested_fix}. The tools are plain functions with
JSON in/out, so an LLM planner (e.g. LangChain tool-calling) can be dropped in by
replacing `run_agent` while keeping the same tools, output schema and evaluation.
"""
from ..config import FAULTS
from ..detector import THR
from .tools import TOOLS, Ctx

FIX = {
    "null_spike": "Check the upstream extract for the affected column (failed join, new optional field, "
                  "or parser change); backfill the day once the source is fixed.",
    "duplicate_rows": "Check for a re-run or double-load of the file; deduplicate on the key columns and add "
                      "an idempotent load (load_date + primary key).",
    "schema_change": "Compare the producer's schema release notes; map the renamed column or restore the "
                     "dropped one, and add a schema contract check before load.",
    "distribution_drift": "Confirm whether the shift is a real business change or an upstream mapping bug; "
                          "if real, retrain/re-baseline, otherwise fix the source transformation.",
    "missing_partial_day": "Check that the file is complete (upstream job finished, no truncated transfer); "
                           "re-pull the day and add a row-count completeness gate.",
    "unit_change": "A column was rescaled (e.g. miles -> km); confirm units with the producer, convert back "
                   "or version the column, and add a unit/range check.",
}
assert set(FIX) == set(FAULTS)


def run_agent(ctx: Ctx, alert: dict):
    trace = []

    def call(tool, **kw):
        res = TOOLS[tool](ctx, **kw)
        trace.append({"step": len(trace) + 1, "tool": tool, "args": kw, "result": res})
        return res

    def done(cause, conf, evidence):
        out = {"cause": cause, "evidence": evidence, "confidence": round(conf, 2), "suggested_fix": FIX[cause]}
        return out, trace

    flags = alert["flags"]

    # 1. schema
    sc = call("compare_schema")
    if sc["changed"]:
        kind = "renamed" if sc["possible_rename"] else "dropped/added"
        return done("schema_change", 0.95,
                    [f"compare_schema: columns removed {sc['removed']}, added {sc['added']} ({kind})"])

    # 2. duplicates
    dk = call("count_duplicate_keys")
    base = max(0.015, 5 * (dk["history_median_dup_rate"] or 0))
    if dk["dup_rate"] > base:
        s = call("sample_bad_rows", kind="duplicate")
        return done("duplicate_rows", 0.92,
                    [f"count_duplicate_keys: {dk['duplicate_rows']} duplicate rows ({dk['dup_rate']:.1%}) on keys "
                     f"{dk['key_columns']}, history median {dk['history_median_dup_rate']}",
                     f"sample_bad_rows: {s['count']} rows share a key with another row"])

    # 3. volume drop
    rc = call("get_metric_history", feature="row_count")
    if rc["z"] is not None and rc["z"] < -THR:
        fb = call("find_first_bad_day", feature="row_count")
        return done("missing_partial_day", 0.9,
                    [f"get_metric_history: row_count {rc['today']:.0f} vs baseline {rc['baseline_median']:.0f} "
                     f"({rc['ratio']:.0%}, z={rc['z']})",
                     f"find_first_bad_day: bad since {fb['first_bad_day']} ({fb['consecutive_bad_days']} day(s))"])

    # 4. null spike
    for f in [f["feature"] for f in flags if f["feature"].endswith(".null_rate")]:
        h = call("get_metric_history", feature=f)
        if h["today"] is not None and h["baseline_median"] is not None and h["today"] - h["baseline_median"] > 0.02:
            col = f.split(".")[0]
            s = call("sample_bad_rows", kind="null", column=col)
            return done("null_spike", 0.9,
                        [f"get_metric_history: {f} {h['today']:.1%} vs baseline {h['baseline_median']:.1%} (z={h['z']})",
                         f"sample_bad_rows: {s['count']} rows with null {col}"])

    # 5. distribution: unit change vs drift
    cols = []
    for f in flags:
        feat = f["feature"]
        col = feat.split(".")[0]
        if col in ("row_count", "dup_rate", "schema") or feat.endswith(".null_rate") or col in cols:
            continue
        cols.append(col)
    evidence, strong = [], False
    for col in cols:
        cd = call("compare_distribution", column=col)
        if "error" in cd:
            continue
        if "quantile_ratio_cv" in cd:
            ev = (f"compare_distribution({col}): KS={cd['ks']}, quantile ratios {cd['quantile_ratios']} "
                  f"(spread cv={cd['quantile_ratio_cv']})")
            if cd["quantile_ratio_cv"] < 0.05 and abs(cd["median_ratio"] - 1) > 0.15:
                return done("unit_change", 0.85,
                            [ev, f"every quantile is scaled by ~{cd['median_ratio']}x: the whole column was "
                                 f"rescaled, not reshaped"])
            strong = strong or cd["ks"] > 0.08
        else:
            shifts = cd["largest_shifts"][0] if cd["largest_shifts"] else {}
            ev = f"compare_distribution({col}): PSI={cd['psi']}, largest shift {shifts}"
            strong = strong or (cd["psi"] or 0) > 0.2
        evidence.append(ev)
    if evidence:
        return done("distribution_drift", 0.8 if strong else 0.5, evidence)
    return done("distribution_drift", 0.3, ["no tool isolated a specific cause; defaulting to drift"])
