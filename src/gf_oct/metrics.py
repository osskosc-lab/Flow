from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Config

REQUIRED_COLUMNS = [
    "episode_id","step","model_tier","operator","prompt_policy","policy_type",
    "q_macro_before","q_macro_after","q_micro_before","q_micro_after",
    "q_total_before","q_total_after","credit_cost","reached_target","terminal",
]

def validate_transitions(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if (df["credit_cost"] < 0).any():
        raise ValueError("Negative credit cost")
    for c in [x for x in REQUIRED_COLUMNS if x.startswith("q_")]:
        if ((df[c] < 0) | (df[c] > 1)).any():
            raise ValueError(f"{c} outside [0,1]")

def summarize_episodes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = []
    for eid, g in df.groupby("episode_id", sort=False):
        g = g.sort_values("step")
        hits = g[g["q_total_after"] >= cfg.target_quality]
        reached = len(hits) > 0
        time = int(hits.iloc[0]["step"]) if reached else int(cfg.max_steps)
        q0 = float(g.iloc[0]["q_total_before"]); qT = float(g.iloc[-1]["q_total_after"])
        qM = float(g.iloc[-1]["q_macro_after"]); qm = float(g.iloc[-1]["q_micro_after"])
        credit = float(g["credit_cost"].sum())
        eta = (qT-q0)/credit if credit > 0 else np.nan
        terminal_loss = cfg.lambda_macro*(1-qM) + cfg.lambda_micro*(1-qm)
        out.append({"episode_id":eid,"world":g.iloc[0].get("world","observed"),
                    "model_tier":g.iloc[0]["model_tier"],"policy_type":g.iloc[0]["policy_type"],
                    "completion_step":time,"event":int(reached),"q_initial":q0,"q_final":qT,
                    "macro_final":qM,"micro_final":qm,"credit_total":credit,
                    "eta_episode":eta,"J":credit+terminal_loss})
    return pd.DataFrame(out)

def rmst(times, events, tau: int) -> float:
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    event_times = np.sort(np.unique(times[(events == 1) & (times <= tau)]))
    survival = 1.0; last = 0.0; area = 0.0
    for t in event_times:
        area += survival*(t-last)
        risk = np.sum(times >= t); d = np.sum((times == t) & (events == 1))
        if risk:
            survival *= (1.0 - d/risk)
        last = t
    return float(area + survival*(tau-last))
