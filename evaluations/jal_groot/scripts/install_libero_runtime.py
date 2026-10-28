"""Install only simulator dependencies inside a derived container image.

The official Isaac simulator setup uses a separate Torch 2.9 environment. Here
we retain each inference image's Torch and preprocessing stack and omit training
packages unused by LiberoEnv. Existing model packages must remain byte-for-byte
version-identical. OpenCV comes from the existing opencv-python-headless wheel.
"""
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

SIMULATOR = {
    "robosuite": "1.4.0", "mujoco": "3.3.1", "bddl": "1.0.1",
    "easydict": "1.9", "gym": "0.25.2", "gym-notices": "0.1.0",
    "numba": "0.65.1", "llvmlite": "0.47.0", "PyOpenGL": "3.1.9",
    "glfw": "2.9.0",
}
SUPPORT_IF_MISSING = {
    "absl-py": "2.2.2", "etils": "1.12.2", "importlib-resources": "6.5.2",
    "zipp": "3.23.0", "fsspec": "2025.3.0", "cloudpickle": "3.1.2",
    "future": "1.0.0", "networkx": "3.4.2",
}
PROTECTED = ("torch", "torchvision", "numpy", "transformers", "tokenizers",
             "huggingface-hub", "safetensors", "pillow", "scipy",
             "opencv-python-headless", "tensorrt", "tensorrt-cu13",
             "nvidia-modelopt")

def versions():
    return {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in metadata.distributions() if d.metadata.get('Name')}

before = versions()
requirements = [f'{name}=={version}' for name, version in SIMULATOR.items()]
requirements += [f'{name}=={version}' for name, version in SUPPORT_IF_MISSING.items()
                 if name.lower().replace('_', '-') not in before]
env = dict(os.environ)
env.pop('PIP_CONSTRAINT', None)
# No dependency resolver is allowed to upgrade the model/preprocessor packages.
if importlib.util.find_spec('pip') is not None:
    installer = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                 '--disable-pip-version-check']
else:
    uv = shutil.which('uv') or '/root/.local/bin/uv'
    installer = [uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                 '--index-url', 'https://pypi.org/simple']
subprocess.run([*installer, *requirements], check=True, env=env)
after = versions()
changed = {name: {'before': before.get(name), 'after': after.get(name)}
           for name in PROTECTED if before.get(name) != after.get(name)}
if changed:
    raise RuntimeError(f'Inference environment changed: {changed}')
report = {'protected_packages': {name: after[name] for name in PROTECTED if name in after},
          'simulator_packages': {name: after[name.lower().replace('_', '-')]
                                 for name in SIMULATOR},
          'all_packages': after,
          'opencv_provider': 'existing opencv-python-headless; no opencv replacement',
          'source_recipe': 'Isaac-GR00T 9c7e746 gr00t/eval/sim/LIBERO/setup_libero.sh'}
Path('/opt/libero/runtime-packages.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['protected_packages'], indent=2))
