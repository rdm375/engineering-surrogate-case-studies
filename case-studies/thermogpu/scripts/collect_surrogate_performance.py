#!/usr/bin/env python3
"""Benchmark the empirically selected exact native-DAG policy at each operating point."""
from __future__ import annotations
import argparse, csv, json, statistics, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'case-studies/thermogpu/raw-results/surrogate-performance'; BUDGETS=(16,32,64,128,256); BATCHES=(1,10,100,1000,10000,100000,1000000)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--budgets',nargs='+',type=int,default=list(BUDGETS)); p.add_argument('--batches',nargs='+',type=int,default=list(BATCHES)); p.add_argument('--dtype',default='float32',choices=['float32','float64']); p.add_argument('--repeats',type=int,default=7); p.add_argument('--warmup',type=int,default=2); p.add_argument('--seed',type=int,default=20261005); p.add_argument('--cpwa-relu-root',type=Path,default=Path('/nvme/Sync/cpwa-relu')); a=p.parse_args()
 sys.path.insert(0,str(a.cpwa_relu_root/'src')); import jax, jax.numpy as jnp
 from cpwa_relu import load_native_dag
 from cpwa_relu.calibration import DAGHardwareProfile
 from cpwa_relu.execution_policy import select_dag_execution_policy
 if a.dtype=='float64': jax.config.update('jax_enable_x64',True)
 device=jax.devices()[0]; jd=getattr(jnp,a.dtype); rng=np.random.default_rng(a.seed); rows=[]
 for b in a.budgets:
  artifact=OUT/f'budget-{b}/artifacts/methane-z-b{b}-native.cpwa'; dag=load_native_dag(artifact)
  for batch in a.batches:
   profile_path=OUT/f'budget-{b}/calibration/{a.dtype}-b{batch}.json'; profile=DAGHardwareProfile.load(profile_path); policy=select_dag_execution_policy(profile); exe=policy.make_executable(dag,dtype=a.dtype,device=device); fn=lambda z,e=exe:e.function(e.params,z)
   x=jax.device_put(jnp.asarray(rng.random((batch,2)),dtype=jd),device); t=time.perf_counter(); y=fn(x); y.block_until_ready(); first=time.perf_counter()-t
   for _ in range(a.warmup): y=fn(x); y.block_until_ready()
   ts=[]
   for _ in range(a.repeats): t=time.perf_counter(); y=fn(x); y.block_until_ready(); ts.append(time.perf_counter()-t)
   med=statistics.median(ts); ns=med*1e9/batch; row={'budget':b,'dtype':a.dtype,'batch':batch,'policy':policy.name,'median_seconds':med,'ns_per_eval':ns,'evaluations_per_second':batch/med,'first_seconds':first,'repeats':a.repeats,'warmup':a.warmup,'backend':jax.default_backend(),'device':str(device),'profile':str(profile_path.relative_to(ROOT))}; rows.append(row); print(f"budget={b:3d} batch={batch:7d} {policy.name:28s} {ns:12.3f} ns/eval")
 path=OUT/f'benchmark-{a.dtype}.csv'; path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='\n') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 (OUT/f'benchmark-{a.dtype}.json').write_text(json.dumps({'status':'ok','rows':rows},indent=2)+'\n'); print(path)
if __name__=='__main__': main()
