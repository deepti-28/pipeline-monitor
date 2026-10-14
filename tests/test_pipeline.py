"""Framework-free tests: python -m pytest tests/test_pipeline.py"""
import numpy as np
import pytest

from backend import pipeline
from backend.config import FAULTS
from backend.detector import detect_robust
from backend.inject_faults import inject
from backend.metrics import compute_snapshot
from backend.synth import make_day


@pytest.fixture(scope="module", autouse=True)
def fresh_state():
    pipeline.ensure_ready()
    pipeline.reset()


def test_clean_day_is_not_flagged():
    r = pipeline.run_next_day()
    assert r["anomaly"] is False


@pytest.mark.parametrize("fault", FAULTS)
def test_each_fault_is_detected_and_diagnosed(fault):
    pipeline.queue_fault(fault)
    r = pipeline.run_next_day()
    assert r["anomaly"], f"{fault} was not detected"
    assert r["alert"]["cause"] == fault
    assert set(r["alert"]) >= {"cause", "evidence", "confidence", "suggested_fix"}


def test_inject_rejects_unknown_fault():
    df = make_day("2024-06-01", np.random.default_rng(0))
    with pytest.raises(ValueError):
        inject(df, "nope", np.random.default_rng(0))


def test_snapshot_survives_dropped_column():
    df = make_day("2024-06-01", np.random.default_rng(0)).drop(columns=["fare_amount"])
    snap = compute_snapshot(df, "2024-06-01")
    assert snap["columns"]["fare_amount"]["null_rate"] == 1.0
