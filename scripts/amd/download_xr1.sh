#!/usr/bin/env bash
# Download pinned XR-1 deployment or foundation weights into the project cache.
set -euo pipefail
FLASHRT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XR1_PYTHON="${FLASHRT_XR1_ENV:-${FLASHRT_ROOT}/.venv-xr1-runtime}/bin/python"
XR1_KIND="${1:-robocasa}"
case "${XR1_KIND}" in
  robocasa)
    "${XR1_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
      XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa \
      --revision 82b7c2210c1a39fca4a87e681cffbbed134ccde0 \
      --local-dir "${FLASHRT_ROOT}/.cache/xr1/robocasa"
    ;;
  5b)
    "${XR1_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
      XiaomiRobotics/Xiaomi-Robotics-1-5B \
      --revision ee21d524b5c52ac961d941e1bc7d6d92836c3d5e \
      --local-dir "${FLASHRT_ROOT}/.cache/xr1/5b"
    "${XR1_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
      Qwen/Qwen3-VL-4B-Instruct \
      --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
      --include '*.json' '*.txt' '*.jinja' \
      --local-dir "${FLASHRT_ROOT}/.cache/xr1/qwen3vl-tokenizer"
    ;;
  demo)
    "${XR1_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
      XiaomiRobotics/xr1_post_train_demo --repo-type dataset \
      --revision e47cda40a259eb5b2cff49b0e8d398b9f80848ee \
      --include 'json/json1.json' 'videos/json1_*.mp4' \
      --local-dir "${FLASHRT_ROOT}/.cache/xr1/demo"
    ;;
  *) echo 'Usage: download_xr1.sh [robocasa|5b|demo]' >&2; exit 2 ;;
esac
