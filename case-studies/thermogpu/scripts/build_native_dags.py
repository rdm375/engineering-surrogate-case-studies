#!/usr/bin/env python3
"""Compile frozen TDAR methane-Z artifacts to exact CPWA-ReLU native DAGs."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from engineering_case_studies.collection import run_capture, slurm_metadata, write_json
from engineering_case_studies.provenance import git_state

ROOT=Path(__file__).resolve().parents[3]
TDAR_RAW=ROOT/'case-studies/thermogpu/raw-results/surrogates/artifacts'
OUT=ROOT/'case-studies/thermogpu/raw-results/surrogate-performance'
BUDGETS=(16,32,64,128,256)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--budgets',nargs='+',type=int,default=list(BUDGETS)); p.add_argument('--cpwa-relu-root',type=Path,default=Path('/nvme/Sync/cpwa-relu')); a=p.parse_args()
 driver=a.cpwa_relu_root/'benchmarks/build_native_dag_artifact.py'
 if not driver.exists(): raise SystemExit(f'missing CPWA-ReLU driver: {driver}')
 env=os.environ.copy(); env['PYTHONPATH']=str(a.cpwa_relu_root/'src')+os.pathsep+env.get('PYTHONPATH','')
 for b in a.budgets:
  src=TDAR_RAW/f'methane-z-b{b}.npz'; d=OUT/f'budget-{b}'; art=d/'artifacts'; art.mkdir(parents=True,exist_ok=True); dst=art/f'methane-z-b{b}-native.cpwa'
  if not src.exists(): raise SystemExit(f'missing TDAR artifact: {src}')
  run=run_capture([sys.executable,str(driver),'--source',str(src),'--output',str(dst)],cwd=a.cpwa_relu_root,env=env)
  (d/'build-native.stdout.txt').write_text(run['stdout']); (d/'build-native.stderr.txt').write_text(run['stderr'])
  rec={'status':'ok' if run['returncode']==0 else 'failed','budget':b,'source':str(src.relative_to(ROOT)),'artifact':str(dst.relative_to(ROOT)),'wall_seconds':run['elapsed_seconds'],'cpwa_relu':git_state(a.cpwa_relu_root),'slurm':slurm_metadata(),'command':run['command']}
  write_json(d/'build-native.json',rec)
  if run['returncode']: raise SystemExit(f'native DAG build failed for budget {b}')
  print(json.dumps(rec,sort_keys=True))
if __name__=='__main__': main()
