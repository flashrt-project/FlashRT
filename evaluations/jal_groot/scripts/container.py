#!/usr/bin/env python3
"""Launch an isolated comparison worker; never stops other containers or changes clocks."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FLASH = 'ghcr.io/flashrt-project/flashrt-thor@sha256:87ad7bce8635207425ac5e90a61d3d83c0ba1042fa28eef7d067474fee872c5c'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runtime', choices=['jal', 'flashrt'])
p.add_argument('--cpu-only', action='store_true')
p.add_argument('--name')
p.add_argument('arguments', nargs=argparse.REMAINDER)
a = p.parse_args()
argv = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
image = FLASH if a.runtime == 'flashrt' else 'groot-comparison-jal:local'
python = '/opt/reference/groot-venv/bin/python' if a.runtime == 'flashrt' else '/opt/gr00t-venv/bin/python'
paths = '/comparison/vendor/Isaac-GR00T:/comparison'
if a.runtime == 'flashrt': paths += ':/opt/FlashRT'
cmd = ['docker', 'run', '--rm', '--network=host', '--shm-size=8g', '-v', f'{ROOT}:/comparison', '-w', '/comparison', '-e', f'PYTHONPATH={paths}', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', f'HF_HOME=/comparison/artifacts/cache/{a.runtime}/hf', '-e', f'XDG_CACHE_HOME=/comparison/artifacts/cache/{a.runtime}', '--entrypoint', python]
if not a.cpu_only:cmd += ['--runtime=nvidia', '--gpus', 'all']
if a.name:cmd += ['--name', a.name]
raise SystemExit(subprocess.run(cmd + [image] + argv).returncode)
