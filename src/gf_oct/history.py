from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import Config

def _augment(df):
    d=df.sort_values(["episode_id","step"]).copy()
    d["delta_total"]=d.q_total_after-d.q_total_before
    d["prev_delta_total"]=d.groupby("episode_id")["delta_total"].shift(1).fillna(0.0)
    d["prev2_delta_total"]=d.groupby("episode_id")["delta_total"].shift(2).fillna(0.0)
    d["prev_operator"]=d.groupby("episode_id")["operator"].shift(1).fillna("START")
    return d

def _matrix(frame, cols):
    return pd.get_dummies(frame[cols], drop_first=False, dtype=float)

def test_h6(df: pd.DataFrame, cfg: Config, rng) -> dict:
    d=_augment(df)
    ids=np.array(d.episode_id.unique(), dtype=object); rng.shuffle(ids)
    split=max(1,int(0.70*len(ids))); train_ids=set(ids[:split])
    d["_set"]=np.where(d.episode_id.isin(train_ids),"train","test")
    markov=["q_macro_before","q_micro_before","q_total_before","credit_cost","operator","model_tier"]
    hist=markov+["prev_delta_total","prev2_delta_total","prev_operator"]
    XM=_matrix(d,markov); XH=_matrix(d,hist)
    train=(d._set=="train").to_numpy(); test=(d._set=="test").to_numpy(); y=d.q_total_after.to_numpy()
    m=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.05,random_state=cfg.seed)
    h=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.05,random_state=cfg.seed)
    m.fit(XM[train],y[train]); h.fit(XH[train],y[train])
    pm=m.predict(XM[test]); ph=h.predict(XH[test])
    evald=d.loc[test,["episode_id"]].copy(); evald["m2"]=(y[test]-pm)**2; evald["h2"]=(y[test]-ph)**2
    rmse_m=float(np.sqrt(evald.m2.mean())); rmse_h=float(np.sqrt(evald.h2.mean())); ratio=rmse_h/rmse_m
    by=evald.groupby("episode_id")[["m2","h2"]].mean().reset_index(); ratios=[]
    for _ in range(cfg.bootstrap_B):
        b=by.sample(len(by),replace=True,random_state=int(rng.integers(2**31-1)))
        ratios.append(float(np.sqrt(b.h2.mean())/np.sqrt(b.m2.mean())))
    ci=[float(np.quantile(ratios,.025)),float(np.quantile(ratios,.975))]
    dec="SUPPORTED" if ci[1] <= cfg.h6_ratio else ("FALSIFIED" if ci[0] >= 1.0 else "INCONCLUSIVE")
    return {"hypothesis":"H6","metric":"RMSE_history_over_markov","rmse_markov":rmse_m,
            "rmse_history":rmse_h,"ratio":ratio,"ci95":ci,"decision":dec}
