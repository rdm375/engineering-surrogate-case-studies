#!/usr/bin/env python3
"""Collect CPWA-ReLU surrogate performance for one budget/batch point.

The benchmark command is intentionally supplied explicitly because CPWA-ReLU
has several released execution policies.  The case study records the exact
command rather than silently choosing a compiler/runtime policy.
"""
from __future__ import annotations
import argparse, json, os, shlex
from pathlib import Path
from engineering_case_studies.collection import parse_performance, run_capture, slurm_metadata, write_json
from engineering_case_studies.provenance import git_state
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/"case-studies/thermogpu/raw-results/surrogate-performance"
STUDY="thermogpu-methane-z-v1"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget",type=int,required=True,choices=[16,32,64,128,256])
    p.add_argument("--batch",type=int,required=True)
    p.add_argument("--cpwa-relu-root",type=Path,default=Path("/nvme/Sync/cpwa-relu"))
    p.add_argument("--command",help="benchmark command template; {budget}, {batch}, {root}, {study_root} are substituted")
    args=p.parse_args()
    template=args.command or os.environ.get("CPWA_RELU_BENCHMARK_COMMAND")
    if not template:
        raise SystemExit("No CPWA-ReLU benchmark command configured. Pass --command or set CPWA_RELU_BENCHMARK_COMMAND; this prevents the case study from silently choosing an execution policy.")
    vals={"budget":args.budget,"batch":args.batch,"root":str(args.cpwa_relu_root),"study_root":str(ROOT)}
    cmd=shlex.split(template.format(**vals))
    run=run_capture(cmd,cwd=args.cpwa_relu_root)
    stem=f"budget-{args.budget}-b{args.batch}"; RAW.mkdir(parents=True,exist_ok=True)
    (RAW/f"{stem}.stdout.txt").write_text(run["stdout"]); (RAW/f"{stem}.stderr.txt").write_text(run["stderr"])
    if run["returncode"]!=0:
        write_json(RAW/f"{stem}.json",{"status":"failed",**run,"slurm":slurm_metadata()}); raise SystemExit("CPWA-ReLU benchmark failed")
    try: ns,rate=parse_performance(run["stdout"]+"\n"+run["stderr"],args.batch)
    except ValueError as e:
        write_json(RAW/f"{stem}.json",{"status":"unparsed",**run,"slurm":slurm_metadata()}); raise SystemExit(str(e))
    record={"study":STUDY,"kind":"performance","method":"cpwa-relu","backend":"gpu","budget":args.budget,"batch_size":args.batch,"ns_per_eval":ns,"evaluations_per_second":rate,"wall_seconds":run["elapsed_seconds"],"source":"CPWA-ReLU","cpwa_relu":git_state(args.cpwa_relu_root),"slurm":slurm_metadata(),"command":run["command"]}
    write_json(RAW/f"{stem}.json",record); print(json.dumps(record,sort_keys=True))
if __name__=="__main__": main()
