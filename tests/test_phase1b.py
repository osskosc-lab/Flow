import json
from pathlib import Path

import pandas as pd
import pytest

from gf_oct.phase1b import run, validate_real_flow


def base_df():
    return pd.DataFrame([{
        "episode_id": "E1",
        "step": 1,
        "model_tier": "high",
        "operator": "generate",
        "prompt_policy": "none",
        "policy_type": "heuristic",
        "q_macro_before": 0.0,
        "q_macro_after": 0.5,
        "q_micro_before": 0.0,
        "q_micro_after": 0.5,
        "q_total_before": 0.0,
        "q_total_after": 0.5,
        "credit_cost": 10.0,
        "reached_target": 0,
        "terminal": 0,
        "source": "real_google_flow",
        "model_name": "test-model",
        "controller_id": "controller-v1",
        "evaluator_id": "evaluator-v1",
        "timestamp_utc": "2026-08-11T00:00:00Z",
        "artifact_uri": "artifact://E1/1",
    }])


def test_real_flow_provenance_accepts_valid_row():
    validate_real_flow(base_df())


def test_real_flow_provenance_rejects_synthetic():
    df = base_df()
    df.loc[0, "source"] = "synthetic"
    with pytest.raises(ValueError):
        validate_real_flow(df)


def test_missing_dataset_yields_waiting_state(tmp_path):
    prereg = Path("preregistration/phase1b.freeze.json")
    out = tmp_path / "out"
    result = run(str(prereg), str(tmp_path / "missing.csv"), str(out))
    assert result["status"] == "WAITING_FOR_REAL_DATA"
    status = json.loads((out / "execution_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "WAITING_FOR_REAL_DATA"
