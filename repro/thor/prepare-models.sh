#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?Usage: prepare-models.sh pi05|groot}"
MODELS="${MODELS:-$PWD/models}"
REF="${REF:-/opt/reference}"
mkdir -p "$MODELS"
MODELS="$(cd "$MODELS" && pwd)"
case "$MODE" in
pi05)
  "$REF/openpi-venv/bin/python" -m pip install --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" google-crc32c
  "$REF/openpi-venv/bin/python" "$ROOT/download_openpi.py" --out "$MODELS/openpi-source"
  "$REF/openpi-venv/bin/python" "$REF/openpi/examples/convert_jax_model_to_pytorch.py" --checkpoint-dir "$MODELS/openpi-source/source/pi05_libero" --config-name pi05_libero --output-path "$MODELS/pi05-openpi" --precision bfloat16
  mkdir -p "$MODELS/pi05-openpi/assets"
  cp -a "$MODELS/openpi-source/source/pi05_libero/assets/." "$MODELS/pi05-openpi/assets/"
  ;;
groot)
  python3 -m venv "$MODELS/download-venv"
  "$MODELS/download-venv/bin/pip" install --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" modelscope==1.40.0 modelscope-hub==0.4.2 requests
  for model in cosmos groot; do
    mapfile -t spec < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]];print(d["repo"]);print(d["revision"])' "$ROOT/versions.json" "$model")
    args=(); [[ "$model" != groot ]] || args=(--subfolder libero_10 --inference-only)
    "$MODELS/download-venv/bin/python" "$ROOT/modelscope_download.py" --repo "${spec[0]}" --revision "${spec[1]}" --out "$MODELS/${spec[0]##*/}" "${args[@]}"
  done
  ;;
*) echo 'Choose pi05 or groot' >&2; exit 2;;
esac
