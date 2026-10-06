#!/usr/bin/env python3
"""Build and validate one frozen TDAR methane-Z surrogate budget."""
from __future__ import annotations
import argparse, csv, json, shutil, sys
from pathlib import Path
from engineering_case_studies.collection import run_capture, slurm_metadata, write_json
from engineering_case_studies.provenance import git_state

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/"case-studies/thermogpu/raw-results/surrogates"
STUDY="thermogpu-methane-z-v1"

def first_row(path: Path):
    with path.open(newline="") as f: return next(csv.DictReader(f))

def pick(row,*names):
    for n in names:
        if n in row and row[n] not in (None,""): return row[n]
    raise KeyError(f"none of {names} present; columns={sorted(row)}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget",type=int,required=True,choices=[16,32,64,128,256])
    p.add_argument("--tdar-root",type=Path,default=Path("/nvme/Sync/tdar"))
    p.add_argument("--thermogpu-root",type=Path,default=Path("/nvme/Sync/ThermoGPU"))
    p.add_argument("--validation-points",type=int,default=100000)
    p.add_argument("--seed",type=int,default=20261002)
    args=p.parse_args()
    oracle=args.thermogpu_root/"build/thermogpu_tdar_oracle"
    driver=args.tdar_root/"benchmarks/run_thermogpu_tradeoff.py"
    out=RAW/f"budget-{args.budget}"
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(driver),"--oracle",str(oracle),"--budgets",str(args.budget),"--initial","16","--validation-points",str(args.validation_points),"--seed",str(args.seed),"--output",str(out)]
    env=dict(__import__('os').environ); env["PYTHONPATH"]=str(args.tdar_root/"src") + (":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    run=run_capture(cmd,cwd=args.tdar_root,env=env)
    (out/"collector.stdout.txt").write_text(run["stdout"]); (out/"collector.stderr.txt").write_text(run["stderr"])
    if run["returncode"]!=0:
        write_json(out/"collector.json",{"status":"failed",**run,"slurm":slurm_metadata()}); raise SystemExit("TDAR tradeoff driver failed")
    summaries=list(out.rglob("summary.csv"))
    if not summaries: raise SystemExit(f"TDAR completed but no summary.csv found under {out}")
    row=first_row(summaries[0])
    record={"study":STUDY,"kind":"surrogate_accuracy","method":"tdar-simplicial-cpwa","budget":args.budget,"rmse":float(pick(row,"rmse","RMSE")),"p99_abs":float(pick(row,"p99_abs","p99","p99_absolute_error")),"linf":float(pick(row,"linf","max_abs","max_absolute_error")),"validation_points":args.validation_points,"seed":args.seed,"source":"TDAR","summary":str(summaries[0].relative_to(ROOT)),"wall_seconds":run["elapsed_seconds"],"tdar":git_state(args.tdar_root),"thermogpu":git_state(args.thermogpu_root),"slurm":slurm_metadata(),"command":run["command"]}
    # Preserve useful complexity/build fields without depending on exact released column names.
    for k in ("simplices","points","sample_count","oracle_evaluations","build_seconds","construction_seconds"):
        if k in row and row[k] not in (None,""):
            try: record[k]=float(row[k]) if "." in row[k] else int(row[k])
            except ValueError: record[k]=row[k]
    write_json(out/"normalized.json",record)
    print(json.dumps(record,sort_keys=True))
if __name__=="__main__": main()
