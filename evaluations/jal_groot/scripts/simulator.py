#!/usr/bin/env python3
"""Launch the isolated LIBERO layer for one inference backend."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runtime', choices=['jal', 'flashrt'])
p.add_argument('--cpu-only', action='store_true')
p.add_argument('--name')
a, argv = p.parse_known_args()
argv = argv[1:] if argv[:1] == ['--'] else argv
python = '/opt/reference/groot-venv/bin/python' if a.runtime == 'flashrt' else '/opt/gr00t-venv/bin/python'
image = f'groot-comparison-{a.runtime}-libero:local'
cmd = ['docker', 'run', '--rm', '--network=none', '--shm-size=8g',
       '-v', f'{ROOT}:/comparison', '-w', '/comparison',
       '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
       '-e', f'HF_HOME=/comparison/artifacts/cache/{a.runtime}/hf',
       '-e', f'XDG_CACHE_HOME=/comparison/artifacts/cache/{a.runtime}',
       '--entrypoint', python]
if a.cpu_only:
    cmd += ['--runtime=runc', '-e', 'NVIDIA_VISIBLE_DEVICES=void']
else:
    cmd += ['--runtime=nvidia', '--gpus', 'all']
if a.name:
    cmd += ['--name', a.name]
raise SystemExit(subprocess.run(cmd + [image] + argv).returncode)
