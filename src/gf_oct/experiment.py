from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config
from .synthetic import nominal_world, null_world, simulate_factorial
from .metrics import validate_transitions, summarize_episodes
from .stats import test_h1,test_h2,test_h3,test_h4,test_h5
from .history import test_h6

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def overall(results):
    d={r["hypothesis"]:r["decision"] for r in results}
    if d.get("H6")=="FALSIFIED" and d.get("H5")=="FALSIFIED":
        return "CORE_CHALLENGED"
    if d.get("H6")=="SUPPORTED" and d.get("H5")=="SUPPORTED":
        return "CORE_SURVIVES_STRONG_TEST"
    return "INCONCLUSIVE"

def analyze(df, cfg, world_name):
    validate_transitions(df)
    ep=summarize_episodes(df,cfg)
    rng=np.random.default_rng(cfg.seed + 991)
    results=[test_h1(ep,cfg,rng),test_h2(ep,cfg,rng),test_h3(df,cfg,rng),
             test_h4(df,cfg,rng),test_h5(ep,cfg,rng),test_h6(df,cfg,rng)]
    return ep,{"world":world_name,"n_episodes":int(ep.episode_id.nunique()),
              "n_transitions":int(len(df)),"results":results,"overall_decision":overall(results)}

def run(prereg_path, out_dir, observed_csv=None):
    cfg=load_config(prereg_path); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    bundle={"theory":"Google Flow Optimal Control Theory","phase":"1A Simulation Falsification",
            "version":"1.0.0","seed":cfg.seed,"prereg_sha256":sha256(prereg_path),"worlds":[]}
    if observed_csv:
        df=pd.read_csv(observed_csv); ep,res=analyze(df,cfg,"observed")
        ep.to_csv(out/"episode_summary_observed.csv",index=False); bundle["worlds"].append(res)
    else:
        for world in (nominal_world(),null_world()):
            df=simulate_factorial(cfg,world); df.to_csv(out/f"transitions_{world.name}.csv",index=False)
            ep,res=analyze(df,cfg,world.name); ep.to_csv(out/f"episode_summary_{world.name}.csv",index=False)
            bundle["worlds"].append(res)
    (out/"decision.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
    for w in bundle["worlds"]:
        for r in w["results"]: rows.append({"world":w["world"],**r})
    pd.DataFrame(rows).to_json(out/"hypothesis_results.json",orient="records",indent=2)
    return bundle
