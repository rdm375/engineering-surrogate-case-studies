#!/usr/bin/env python3
"""Collect direct ThermoGPU performance for one batch size.

For --backend cpu both scalar and OpenMP profiles are collected.  --backend cuda
collects the CUDA profile.  Native stdout/stderr are retained verbatim beside a
normalized JSON record.
"""
from __future__ import annotations
import argparse, json, shlex
from pathlib import Path
from engineering_case_studies.collection import parse_performance, run_capture, slurm_metadata, write_json
from engineering_case_studies.provenance import git_state

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/"case-studies/thermogpu/raw-results/direct"
STUDY="thermogpu-methane-z-v1"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--thermogpu-root",type=Path,required=True)
    p.add_argument("--backend",choices=["cpu","scalar","openmp","cuda"],required=True)
    p.add_argument("--batch",type=int,required=True)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--executable",type=Path)
    args=p.parse_args()
    exe=args.executable or args.thermogpu_root/"build/thermogpu_scaling"
    if not exe.exists(): raise SystemExit(f"ThermoGPU scaling executable not found: {exe}")
    profiles=["scalar","openmp"] if args.backend=="cpu" else [args.backend]
    for profile in profiles:
        # ThermoGPU's released scaling CLI is: --profile PROFILE REPEATS BATCH.
        cmd=[str(exe),"--profile",profile,str(args.repeats),str(args.batch)]
        run=run_capture(cmd,cwd=args.thermogpu_root)
        stem=f"{profile}-b{args.batch}"
        (RAW/f"{stem}.stdout.txt").parent.mkdir(parents=True,exist_ok=True)
        (RAW/f"{stem}.stdout.txt").write_text(run["stdout"])
        (RAW/f"{stem}.stderr.txt").write_text(run["stderr"])
        if run["returncode"] != 0:
            write_json(RAW/f"{stem}.json",{"status":"failed",**run,"slurm":slurm_metadata()})
            raise SystemExit(f"ThermoGPU {profile} benchmark failed ({run['returncode']})")
        try: ns,rate=parse_performance(run["stdout"]+"\n"+run["stderr"],args.batch)
        except ValueError as e:
            write_json(RAW/f"{stem}.json",{"status":"unparsed",**run,"slurm":slurm_metadata()})
            raise SystemExit(f"{e}; native output retained in {RAW}")
        record={"study":STUDY,"kind":"performance","method":f"thermogpu-{profile}","backend":profile,"batch_size":args.batch,"repeats":args.repeats,"ns_per_eval":ns,"evaluations_per_second":rate,"wall_seconds":run["elapsed_seconds"],"source":"ThermoGPU","thermogpu":git_state(args.thermogpu_root),"slurm":slurm_metadata(),"command":run["command"]}
        write_json(RAW/f"{stem}.json",record)
        print(json.dumps(record,sort_keys=True))
if __name__=="__main__": main()
