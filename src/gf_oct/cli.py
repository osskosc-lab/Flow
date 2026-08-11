from __future__ import annotations
import argparse, json
from .experiment import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prereg",default="preregistration/phase1a.freeze.json")
    p.add_argument("--out",default="results/phase1a")
    p.add_argument("--data",default=None,help="Optional real transition CSV; omitted => synthetic controls")
    a=p.parse_args()
    result=run(a.prereg,a.out,a.data)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
