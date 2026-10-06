"""Shared helpers for case-study data collectors."""
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path
from typing import Sequence

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def run_capture(command: Sequence[str], *, cwd: Path | None = None, env: dict[str,str] | None=None) -> dict:
    t0=time.perf_counter()
    p=subprocess.run(list(command), cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed=time.perf_counter()-t0
    return {"command":list(command),"returncode":p.returncode,"elapsed_seconds":elapsed,"stdout":p.stdout,"stderr":p.stderr}

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")

def slurm_metadata() -> dict[str,str]:
    keys=["SLURM_JOB_ID","SLURM_ARRAY_JOB_ID","SLURM_ARRAY_TASK_ID","SLURM_JOB_NODELIST","SLURM_CPUS_PER_TASK","SLURM_GPUS","CUDA_VISIBLE_DEVICES"]
    return {k:os.environ[k] for k in keys if k in os.environ}

def parse_performance(text: str, batch_size: int) -> tuple[float,float]:
    """Parse common ThermoGPU/CPWA benchmark output; return ns/eval and eval/s."""
    patterns=[
        rf"({_FLOAT})\s*ns/eval",
        rf"ns(?:_per_eval|/eval)?\s*[:=]\s*({_FLOAT})",
    ]
    ns=None
    for pat in patterns:
        m=re.search(pat,text,re.I)
        if m: ns=float(m.group(1)); break
    if ns is None:
        # JSON-line / JSON-document fallback
        for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
            try: obj=json.loads(line)
            except Exception: continue
            for key in ("ns_per_eval","ns/eval"):
                if key in obj: ns=float(obj[key]); break
            if ns is not None: break
    if ns is None or ns <= 0:
        raise ValueError("could not parse a positive ns/eval measurement")
    return ns, 1.0e9/ns
