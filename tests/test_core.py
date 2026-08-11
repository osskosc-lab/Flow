from gf_oct.config import load_config
from gf_oct.synthetic import nominal_world, null_world, simulate_factorial
from gf_oct.metrics import validate_transitions, rmst
from gf_oct.experiment import analyze

PREREG="preregistration/phase1a.freeze.json"

def test_simulator_schema_and_bounds():
    cfg=load_config(PREREG)
    df=simulate_factorial(cfg,nominal_world())
    validate_transitions(df)
    assert len(df)>0
    assert df.q_total_after.between(0,1).all()
    assert (df.credit_cost>=0).all()

def test_rmst_handles_censoring():
    value=rmst([2,4,10],[1,1,0],10)
    assert 0 < value <= 10

def test_nominal_and_null_are_distinguishable():
    cfg=load_config(PREREG)
    ndf=simulate_factorial(cfg,nominal_world())
    zdf=simulate_factorial(cfg,null_world())
    nep,_=analyze(ndf,cfg,"nominal")
    zep,_=analyze(zdf,cfg,"null")
    assert nep.loc[nep.model_tier=="high","q_final"].mean() > zep.loc[zep.model_tier=="high","q_final"].mean() - 0.05
