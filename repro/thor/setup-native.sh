#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$PWD/runtime}"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
python "$ROOT/unpack-runtime.py" --out "$WORK/source"
SOURCE="$WORK/source"
bash "$ROOT/install-native.sh" "$SOURCE"
bash "$ROOT/setup-reference-envs.sh" "$WORK/reference"
printf 'export SOURCE=%q\nexport REF=%q\nexport PYTHONPATH=%q\nexport CUTE_DSL_ARCH=sm_101a\nexport LINGBOT_FA4_SRC=%q\n' "$SOURCE" "$WORK/reference" "$SOURCE" "$SOURCE/csrc/attention/flash_attn_4_src" > "$WORK/runtime.env"
echo "Next: source $WORK/runtime.env"
