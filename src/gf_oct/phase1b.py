from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .experiment import analyze
from .metrics import validate_transitions

PROVENANCE_COLUMNS = [
    "source",
    "model_name",
    "controller_id",
    "evaluator_id",
    "timestamp_utc",
    "artifact_uri",
]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_real_flow(df: pd.DataFrame) -> None:
    validate_transitions(df)
    missing = [c for c in PROVENANCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing real-Flow provenance columns: {missing}")
    if not (df["source"].astype(str) == "real_google_flow").all():
        raise ValueError("Phase 1B accepts source=real_google_flow only")
    if df[["episode_id", "step"]].duplicated().any():
        raise ValueError("Duplicate episode_id-step rows detected")
    for c in ["model_name", "controller_id", "evaluator_id", "timestamp_utc", "artifact_uri"]:
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
            raise ValueError(f"Empty provenance values in {c}")


def arm_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "model_high": int((df["model_tier"] == "high").groupby(df["episode_id"]).max().sum()),
        "model_low": int((df["model_tier"] == "low").groupby(df["episode_id"]).max().sum()),
        "delta_prompt": int((df["prompt_policy"] == "delta").groupby(df["episode_id"]).max().sum()),
        "full_prompt": int((df["prompt_policy"] == "full").groupby(df["episode_id"]).max().sum()),
        "learned_policy": int((df["policy_type"] == "learned").groupby(df["episode_id"]).max().sum()),
        "heuristic_policy": int((df["policy_type"] == "heuristic").groupby(df["episode_id"]).max().sum()),
    }


def run(prereg_path: str, data_path: str, out_dir: str, require_complete: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prereg = Path(prereg_path)
    data = Path(data_path)
    cfg = load_config(prereg)

    base = {
        "theory": "Google Flow Optimal Control Theory",
        "phase": "1B Real-Flow Confirmatory Execution",
        "version": "1.0.0",
        "prereg_sha256": sha256(prereg),
        "status": None,
    }

    if not data.exists():
        base.update({
            "status": "WAITING_FOR_REAL_DATA",
            "reason": f"No confirmatory dataset at {data}",
            "next_action": "Collect frozen real Google Flow transitions and commit/upload the CSV without changing preregistration thresholds.",
        })
        (out / "execution_status.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        return base

    df = pd.read_csv(data)
    validate_real_flow(df)
    counts = arm_counts(df)
    n_episodes = int(df["episode_id"].nunique())
    complete = n_episodes >= 120 and min(counts.values()) >= cfg.n_episodes_per_arm

    if require_complete and not complete:
        base.update({
            "status": "DATA_INCOMPLETE",
            "dataset_sha256": sha256(data),
            "n_episodes": n_episodes,
            "arm_counts": counts,
            "required_total_episodes_min": 120,
            "required_per_arm": cfg.n_episodes_per_arm,
        })
        (out / "execution_status.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        return base

    episodes, result = analyze(df, cfg, "real_google_flow")
    episodes.to_csv(out / "episode_summary_real_flow.csv", index=False)
    result.update({
        "dataset_sha256": sha256(data),
        "prereg_sha256": sha256(prereg),
        "arm_counts": counts,
        "collection_complete": complete,
    })
    bundle = {**base, "status": "CONFIRMATORY_ANALYSIS_COMPLETE" if complete else "INTERIM_NONCONFIRMATORY", "result": result}
    (out / "decision.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "execution_status.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="GF-OCT Phase 1B real-Flow confirmatory analysis")
    p.add_argument("--prereg", default="preregistration/phase1b.freeze.json")
    p.add_argument("--data", default="data/phase1b/flow_transitions.csv")
    p.add_argument("--out", default="results/phase1b")
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    result = run(args.prereg, args.data, args.out, args.require_complete)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
