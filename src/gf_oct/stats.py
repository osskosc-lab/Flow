from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Config
from .metrics import rmst

def _ci(vals, alpha=0.05):
    return [float(np.quantile(vals, alpha/2)), float(np.quantile(vals, 1-alpha/2))]

def _boot_diff(x, y, rng, B):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = rng.choice(x, len(x), True).mean() - rng.choice(y, len(y), True).mean()
    return _ci(vals)

def _lower_better(effect, ci, delta):
    if ci[1] <= -delta:
        return "SUPPORTED"
    if ci[0] >= 0:
        return "FALSIFIED"
    return "INCONCLUSIVE"

def test_h1(ep: pd.DataFrame, cfg: Config, rng) -> dict:
    h, l = ep[ep.model_tier=="high"], ep[ep.model_tier=="low"]
    rh, rl = rmst(h.completion_step, h.event, cfg.max_steps), rmst(l.completion_step, l.event, cfg.max_steps)
    vals = []
    for _ in range(cfg.bootstrap_B):
        bh = h.sample(len(h), replace=True, random_state=int(rng.integers(2**31-1)))
        bl = l.sample(len(l), replace=True, random_state=int(rng.integers(2**31-1)))
        vals.append(rmst(bh.completion_step,bh.event,cfg.max_steps)-rmst(bl.completion_step,bl.event,cfg.max_steps))
    ci = _ci(vals); effect = rh-rl
    return {"hypothesis":"H1","metric":"RMST_high_minus_low","effect":effect,"ci95":ci,
            "rmst_high":rh,"rmst_low":rl,"decision":_lower_better(effect,ci,cfg.h1_delta)}

def test_h2(ep: pd.DataFrame, cfg: Config, rng) -> dict:
    h = ep.loc[ep.model_tier=="high","eta_episode"].dropna(); l = ep.loc[ep.model_tier=="low","eta_episode"].dropna()
    effect = float(h.mean()-l.mean()); ci = _boot_diff(h,l,rng,cfg.bootstrap_B)
    dec = "SUPPORTED" if ci[0] >= cfg.h2_delta else ("FALSIFIED" if ci[1] <= 0 else "INCONCLUSIVE")
    return {"hypothesis":"H2","metric":"eta_episode_high_minus_low","effect":effect,"ci95":ci,"decision":dec}

def test_h3(df: pd.DataFrame, cfg: Config, rng) -> dict:
    e = df[df.operator=="edit"].copy(); drift = np.abs(e.q_macro_after-e.q_macro_before).to_numpy()
    vals = [rng.choice(drift,len(drift),True).mean() for _ in range(cfg.bootstrap_B)]
    ci = _ci(vals); effect=float(np.mean(drift))
    dec = "SUPPORTED" if ci[1] <= cfg.h3_margin else ("FALSIFIED" if ci[0] > cfg.h3_margin else "INCONCLUSIVE")
    return {"hypothesis":"H3","metric":"mean_absolute_macro_drift_after_edit","effect":effect,
            "ci95":ci,"margin":cfg.h3_margin,"decision":dec}

def test_h4(df: pd.DataFrame, cfg: Config, rng) -> dict:
    e = df[df.operator=="edit"].copy(); e["drift"] = np.abs(e.q_macro_after-e.q_macro_before)
    d=e.loc[e.prompt_policy=="delta","drift"]; f=e.loc[e.prompt_policy=="full","drift"]
    effect=float(d.mean()-f.mean()); ci=_boot_diff(d,f,rng,cfg.bootstrap_B)
    return {"hypothesis":"H4","metric":"macro_drift_delta_minus_full","effect":effect,"ci95":ci,
            "decision":_lower_better(effect,ci,cfg.h4_delta)}

def test_h5(ep: pd.DataFrame, cfg: Config, rng) -> dict:
    l=ep.loc[ep.policy_type=="learned","J"]; h=ep.loc[ep.policy_type=="heuristic","J"]
    effect=float(l.mean()-h.mean()); ci=_boot_diff(l,h,rng,cfg.bootstrap_B)
    return {"hypothesis":"H5","metric":"J_learned_minus_heuristic","effect":effect,"ci95":ci,
            "decision":_lower_better(effect,ci,cfg.h5_delta)}
