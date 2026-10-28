#!/usr/bin/env python3
"""Three predeclared timing rounds; run only after authorized GPU isolation."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
if sys.argv[1:] != ['--exclusive-gpu']:
    raise SystemExit('Requires --exclusive-gpu after stopping competing workloads and fixing clocks')
orders=[['trt','flashrt'],['flashrt','trt'],['trt','flashrt']]
plan={'orders':orders,'warmup_per_prompt':20,'calls_per_prompt':100,'rounds':3,'started_utc':datetime.now(timezone.utc).isoformat(),'completed':[]}
plan_path=ROOT/'artifacts/latency-plan.json'
if plan_path.exists():raise SystemExit('Latency plan already exists; preserve its evidence')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
for index,order in enumerate(orders,1):
    for backend in order:
        containers=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip()
        if containers:raise SystemExit(f'Competing container(s) before timing: {containers}')
        runtime='flashrt' if backend=='flashrt' else 'jal'
        extra=['--capture-dir','/comparison/artifacts/captures/heldout'] if backend=='flashrt' else ['--engine-dir','/comparison/artifacts/jal_trt_optimized_mixed_nvfp4/engines']
        output=f'/comparison/artifacts/runs/heldout-{backend}-timing-r{index}'
        cmd=[sys.executable,str(ROOT/'scripts/container.py'),runtime,'--','-m','groot_compare.run','run','--backend',backend,'--phase','timing','--manifest','/comparison/artifacts/observations/heldout/manifest.json','--checkpoint','/comparison/artifacts/models/groot-libero-onecam','--output',output,'--exclusive-gpu']+extra
        print(f'Round {index}: {backend}',flush=True)
        with (ROOT/f'artifacts/{backend}-timing-r{index}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        plan['completed'].append({'round':index,'backend':backend,'output':output,'finished_utc':datetime.now(timezone.utc).isoformat()})
        plan_path.write_text(json.dumps(plan,indent=2)+'\n')
print('All timing rounds completed',flush=True)
