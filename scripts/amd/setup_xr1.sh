#!/usr/bin/env bash
# XR-1 inference environment for gfx1151, Python 3.12 and AMD ROCm 7.2 wheels.
set -euo pipefail
FLASHRT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XR1_ENV="${FLASHRT_XR1_ENV:-${FLASHRT_ROOT}/.venv-xr1-runtime}"
XR1_PYTHON="${FLASHRT_XR1_PYTHON:-python3.12}"
"${XR1_PYTHON}" -m venv "${XR1_ENV}"
"${XR1_ENV}/bin/python" -m ensurepip --upgrade
XR1_WHEELS='https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2'
"${XR1_ENV}/bin/python" -m pip install \
    "${XR1_WHEELS}/torch-2.10.0%2Brocm7.2.0.lw.gitb6ee5fde-cp312-cp312-linux_x86_64.whl" \
    "${XR1_WHEELS}/torchvision-0.25.0%2Brocm7.2.0.git82df5f59-cp312-cp312-linux_x86_64.whl" \
    "${XR1_WHEELS}/triton-3.6.0%2Brocm7.2.0.gitba5c1517-cp312-cp312-linux_x86_64.whl"
"${XR1_ENV}/bin/python" -m pip install -e "${FLASHRT_ROOT}[xr1]"
"${XR1_ENV}/bin/python" -m pip install cmake pybind11
PATH="${XR1_ENV}/bin:${PATH}" PYTHON="${XR1_ENV}/bin/python" \
    bash "${FLASHRT_ROOT}/scripts/amd/build_amd.sh" gfx1151
"${XR1_ENV}/bin/python" - <<'PY'
import torch
if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError('ROCm PyTorch GPU is unavailable')
arch = torch.cuda.get_device_properties(0).gcnArchName
if arch.split(':')[0] != 'gfx1151':
    raise RuntimeError(f'Expected gfx1151, found {arch}')
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print('XR-1 environment ready:', torch.__version__, 'HIP', torch.version.hip, arch)
PY
