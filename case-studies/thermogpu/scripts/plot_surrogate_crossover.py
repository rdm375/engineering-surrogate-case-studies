#!/usr/bin/env python3
"""Plot exact CPWA-ReLU throughput versus batch for every TDAR budget."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=ROOT/'case-studies/thermogpu/processed-results/surrogate-performance.csv'); p.add_argument('--output',type=Path,default=ROOT/'case-studies/thermogpu/figures/surrogate-throughput-vs-batch.png'); a=p.parse_args()
 with a.input.open(newline='') as f: rows=list(csv.DictReader(f))
 fig,ax=plt.subplots(figsize=(10.2,6.6))
 for b in sorted({int(r['budget']) for r in rows}):
  q=sorted((r for r in rows if int(r['budget'])==b),key=lambda r:int(r['batch'])); ax.plot([int(r['batch']) for r in q],[float(r['evaluations_per_second'])/1e6 for r in q],marker='o',label=f'budget {b}')
 ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Batch size'); ax.set_ylabel('Million evaluations / second'); ax.set_title('ThermoGPU methane-Z: exact CPWA-ReLU throughput'); ax.grid(True,which='both',alpha=.25); ax.legend(); fig.tight_layout(); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=150); print(a.output)
if __name__=='__main__':main()
