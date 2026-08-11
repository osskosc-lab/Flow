from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    seed: int
    n_episodes_per_arm: int
    max_steps: int
    target_quality: float
    bootstrap_B: int
    alpha: float
    h1_delta: float
    h2_delta: float
    h3_margin: float
    h4_delta: float
    h5_delta: float
    h6_ratio: float
    lambda_macro: float
    lambda_micro: float

def load_config(path: str | Path) -> Config:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    t = d["thresholds"]
    l = d["loss"]
    return Config(
        seed=int(d["seed"]),
        n_episodes_per_arm=int(d["n_episodes_per_arm"]),
        max_steps=int(d["max_steps"]),
        target_quality=float(d["target_quality"]),
        bootstrap_B=int(d["bootstrap_B"]),
        alpha=float(d["alpha"]),
        h1_delta=float(t["H1_min_rmst_advantage_steps"]),
        h2_delta=float(t["H2_min_eta_advantage"]),
        h3_margin=float(t["H3_macro_equivalence_margin"]),
        h4_delta=float(t["H4_min_delta_prompt_drift_reduction"]),
        h5_delta=float(t["H5_min_policy_cost_advantage"]),
        h6_ratio=float(t["H6_history_rmse_ratio_support"]),
        lambda_macro=float(l["lambda_macro"]),
        lambda_micro=float(l["lambda_micro"]),
    )
