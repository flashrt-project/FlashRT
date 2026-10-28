"""CPU-only simulator import check. Does not construct a simulator or model."""
import importlib.metadata as metadata
import json
from pathlib import Path

import torch
import robosuite
import mujoco
from libero.libero import benchmark
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
from groot_compare.closed_loop import _verify_sources

_verify_sources(Path("/comparison/vendor/Isaac-GR00T"),
                Path("/comparison/vendor/Isaac-GR00T/external_dependencies/LIBERO"))

if torch.cuda.is_initialized():
    raise RuntimeError('CUDA initialized during CPU-only simulator import check')
suite = benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
report = json.loads(Path('/opt/libero/runtime-packages.json').read_text())
print(json.dumps({'cpu_only_imports_passed': True, 'cuda_initialized': False,
                  'libero_task_count': suite.get_num_tasks(),
                  'torch': torch.__version__, 'mujoco': mujoco.__version__,
                  'robosuite': metadata.version('robosuite'),
                  'protected_packages': report['protected_packages']}, indent=2))
