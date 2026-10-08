#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD/reference}"
PYTHON="${PYTHON:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SITE="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
TORCH_SITE="$("$PYTHON" -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent.parent)')"
CUTLASS_SITE="$("$PYTHON" -c 'import importlib.util,importlib.metadata,pathlib; s=importlib.util.find_spec("cutlass"); print(pathlib.Path(s.origin).parent.parent if s and s.origin else importlib.metadata.distribution("nvidia-cutlass-dsl").locate_file("nvidia_cutlass_dsl/python_packages"))')"
# Resolve the installed CUDA runtime first, then prevent user/PYTHONPATH
# packages from satisfying dependency installation outside the new environment.
export PYTHONNOUSERSITE=1 GIT_LFS_SKIP_SMUDGE=1
unset PYTHONPATH
MODE="${2:-all}"
case "$MODE" in openpi|groot|all) ;; *) echo "Choose openpi, groot or all" >&2; exit 2;; esac
mkdir -p "$ROOT"
pip_install() {
    local python="$1"
    shift
    if ! "$python" -m pip install --retries 1 --timeout 20 --index-url "$PIP_INDEX_URL" "$@"; then
        [[ "$PIP_INDEX_URL" != https://pypi.org/simple ]] || return 1
        echo "Configured package index failed; retrying official PyPI" >&2
        "$python" -m pip install --index-url https://pypi.org/simple "$@"
    fi
}

fetch_source() {
    local url="$1" dest="$2" revision="$3"
    git init "$dest"
    git -C "$dest" remote get-url origin >/dev/null 2>&1 || git -C "$dest" remote add origin "$url"
    local attempt fetched=0
    for attempt in 1 2 3; do
        if git -C "$dest" -c http.version=HTTP/1.1 fetch --depth 1 origin "$revision"; then
            fetched=1
            break
        fi
        echo "Source fetch failed (attempt $attempt/3)" >&2
    done
    [[ "$fetched" == 1 ]]
    git -C "$dest" checkout --detach FETCH_HEAD
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]]
}

if [[ "$MODE" == groot || "$MODE" == all ]]; then
fetch_source https://github.com/NVIDIA/Isaac-GR00T.git "$ROOT/Isaac-GR00T" 9c7e746b2cd37a810070a98ef41d290a07e806c2
"$PYTHON" -m venv "$ROOT/groot-venv"
SITE="$("$ROOT/groot-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
printf "import sys; sys.path.extend(p for p in ('%s','%s','%s') if p and p not in sys.path)\n" "$BASE_SITE" "$TORCH_SITE" "$CUTLASS_SITE" > "$SITE/cuda-runtime.pth"
pip_install "$ROOT/groot-venv/bin/python" setuptools wheel hatchling editables
pip_install "$ROOT/groot-venv/bin/python" -r "$SCRIPT_DIR/requirements-groot.txt"
pip_install "$ROOT/groot-venv/bin/python" --ignore-requires-python --no-build-isolation --no-deps -e "$ROOT/Isaac-GR00T"
fi
if [[ "$MODE" == openpi || "$MODE" == all ]]; then
fetch_source https://github.com/Physical-Intelligence/openpi.git "$ROOT/openpi" 15a9616a00943ada6c20a0f158e3adb39df2ccac
fetch_source https://github.com/huggingface/lerobot.git "$ROOT/lerobot" 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
"$PYTHON" -m venv "$ROOT/openpi-venv"
SITE="$("$ROOT/openpi-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
printf "import sys; sys.path.extend(p for p in ('%s','%s','%s') if p and p not in sys.path)\n" "$BASE_SITE" "$TORCH_SITE" "$CUTLASS_SITE" > "$SITE/cuda-runtime.pth"
pip_install "$ROOT/openpi-venv/bin/python" setuptools wheel hatchling editables poetry-core
pip_install "$ROOT/openpi-venv/bin/python" -r "$SCRIPT_DIR/requirements-openpi.txt"
pip_install "$ROOT/openpi-venv/bin/python" --no-build-isolation --no-deps -e "$ROOT/openpi" -e "$ROOT/openpi/packages/openpi-client" -e "$ROOT/lerobot"
SITE="$("$ROOT/openpi-venv/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
cp -r "$ROOT/openpi/src/openpi/models_pytorch/transformers_replace/"* "$SITE/transformers/"
fi
echo "Reference environments installed at $ROOT"
