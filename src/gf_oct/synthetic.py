from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import Config

@dataclass(frozen=True)
class World:
    name: str
    high_power: float
    low_power: float
    high_cost: float = 30.0
    low_cost: float = 10.0
    high_noise: float = 0.025
    low_noise: float = 0.050
    edit_micro_gain: float = 0.40
    edit_macro_drift_delta: float = 0.018
    edit_macro_drift_full: float = 0.060
    history_strength: float = 0.10
    learned_policy_bonus: float = 0.08

def nominal_world() -> World:
    return World(name="nominal", high_power=0.58, low_power=0.23, history_strength=0.12, learned_policy_bonus=0.09)

def null_world() -> World:
    return World(name="null", high_power=0.30, low_power=0.30, high_noise=0.04, low_noise=0.04,
                 edit_macro_drift_delta=0.080, edit_macro_drift_full=0.080,
                 history_strength=0.0, learned_policy_bonus=0.0)

def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def simulate_arm(cfg: Config, world: World, model_tier: str, prompt_policy: str,
                 policy_type: str, episode_offset: int, n: int,
                 rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    power = world.high_power if model_tier == "high" else world.low_power
    cost_gen = world.high_cost if model_tier == "high" else world.low_cost
    noise = world.high_noise if model_tier == "high" else world.low_noise
    for ep in range(n):
        eid = f"{world.name}-{episode_offset+ep:05d}"
        qM = _clip(rng.normal(0.20, 0.06)); qm = _clip(rng.normal(0.18, 0.06))
        q = 0.65*qM + 0.35*qm
        prev_gain = 0.0
        for step in range(cfg.max_steps):
            threshold = 0.67 - (world.learned_policy_bonus if policy_type == "learned" else 0.0)
            if q >= cfg.target_quality:
                break
            if qM < threshold:
                op = "generate"
            elif qm < 0.88:
                op = "edit"
            else:
                op = "stop"
            qM0, qm0, q0 = qM, qm, q
            if op == "generate":
                gain = power*(1.0-q)*rng.beta(5,2) + world.history_strength*prev_gain
                qM = _clip(qM + 0.85*gain + rng.normal(0,noise))
                qm = _clip(qm + 0.65*gain + rng.normal(0,noise))
                credit = cost_gen; ppol = "none"
            elif op == "edit":
                drift_sd = world.edit_macro_drift_delta if prompt_policy == "delta" else world.edit_macro_drift_full
                qM = _clip(qM + rng.normal(0,drift_sd))
                micro_gain = world.edit_micro_gain*(1.0-qm)*rng.beta(6,2)
                qm = _clip(qm + micro_gain + rng.normal(0,0.025))
                credit = 40.0; ppol = prompt_policy
                gain = (0.65*qM + 0.35*qm) - q0
            else:
                credit = 0.0; ppol = "none"; gain = 0.0
            q = _clip(0.65*qM + 0.35*qm)
            reached = int(q >= cfg.target_quality)
            terminal = int(reached or op == "stop" or step == cfg.max_steps-1)
            rows.append({"world":world.name,"episode_id":eid,"step":step+1,"model_tier":model_tier,
                         "operator":op,"prompt_policy":ppol,"policy_type":policy_type,
                         "q_macro_before":qM0,"q_macro_after":qM,"q_micro_before":qm0,"q_micro_after":qm,
                         "q_total_before":q0,"q_total_after":q,"credit_cost":credit,
                         "reached_target":reached,"terminal":terminal})
            prev_gain = q-q0
            if terminal:
                break
    return pd.DataFrame(rows)

def simulate_factorial(cfg: Config, world: World) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + (0 if world.name == "nominal" else 1))
    frames = []; offset = 0
    for tier in ("high","low"):
        for prompt_policy in ("delta","full"):
            for policy_type in ("learned","heuristic"):
                frames.append(simulate_arm(cfg,world,tier,prompt_policy,policy_type,offset,
                                           cfg.n_episodes_per_arm,rng))
                offset += cfg.n_episodes_per_arm
    return pd.concat(frames, ignore_index=True)
