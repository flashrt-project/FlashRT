#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD/reference}"
PYTHON="${PYTHON:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SITE="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
TORCH_SITE="$("$PYTHON" -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent.parent)')"
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$ROOT"
fetch_source() {
    local url="$1" dest="$2" revision="$3"
    git init "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch --depth 1 origin "$revision"
    git -C "$dest" checkout --detach FETCH_HEAD
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]]
}

fetch_source https://github.com/NVIDIA/Isaac-GR00T.git "$ROOT/Isaac-GR00T" 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
"$PYTHON" -m venv --system-site-packages "$ROOT/groot-venv"
SITE="$("$ROOT/groot-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
printf "import site,sys; None if getattr(sys,'_jal_cuda_sites',False) else (setattr(sys,'_jal_cuda_sites',True),site.addsitedir('%s'),site.addsitedir('%s'))\n" "$BASE_SITE" "$TORCH_SITE" > "$SITE/cuda-runtime.pth"
"$ROOT/groot-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" setuptools wheel hatchling editables
"$ROOT/groot-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -r "$SCRIPT_DIR/requirements-groot.txt"
"$ROOT/groot-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" --ignore-requires-python --no-deps -e "$ROOT/Isaac-GR00T"
fetch_source https://github.com/Physical-Intelligence/openpi.git "$ROOT/openpi" 15a9616a00943ada6c20a0f158e3adb39df2ccac
fetch_source https://github.com/huggingface/lerobot.git "$ROOT/lerobot" 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
"$PYTHON" -m venv --system-site-packages "$ROOT/openpi-venv"
SITE="$("$ROOT/openpi-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
printf "import site,sys; None if getattr(sys,'_jal_cuda_sites',False) else (setattr(sys,'_jal_cuda_sites',True),site.addsitedir('%s'),site.addsitedir('%s'))\n" "$BASE_SITE" "$TORCH_SITE" > "$SITE/cuda-runtime.pth"
"$ROOT/openpi-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" setuptools wheel hatchling editables
"$ROOT/openpi-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -r "$SCRIPT_DIR/requirements-openpi.txt"
"$ROOT/openpi-venv/bin/python" -m pip install --index-url "$PIP_INDEX_URL" --no-deps -e "$ROOT/openpi" -e "$ROOT/openpi/packages/openpi-client" -e "$ROOT/lerobot"
SITE="$("$ROOT/openpi-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
cp -r "$ROOT/openpi/src/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"
echo "Reference environments installed at $ROOT"
