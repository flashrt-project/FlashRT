#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-help}"; shift || true
cd "$ROOT"
export MODELS="${MODELS:-/models}" OUT="${OUT:-/results}"
case "$COMMAND" in
prepare) exec bash "$ROOT/prepare-models.sh" "${1:?Choose pi05 or groot}";;
validate)
  MODE="${1:?Choose pi05, groot or all}"
  export PI05="${PI05:-$MODELS/pi05-openpi}"
  if [[ "$MODE" == groot || "$MODE" == all ]]; then
    mkdir -p "$OUT"
    OVERLAY="$(mktemp -d "$OUT/groot-overlay.XXXXXX")"; rmdir "$OVERLAY"
    python "$ROOT/local_groot_checkpoint.py" --checkpoint "$MODELS/GR00T-N1.7-LIBERO/libero_10" --cosmos "$MODELS/Cosmos-Reason2-2B" --out "$OVERLAY"
    export GROOT="$OVERLAY"
  fi
  exec bash "$ROOT/run-validation.sh" "$MODE";;
doctor) exec python "$ROOT/doctor.py";;
bash) exec bash "$@";;
*) printf '%s\n' 'Commands: prepare pi05|groot; validate pi05|groot|all; doctor; bash' ;;
esac
