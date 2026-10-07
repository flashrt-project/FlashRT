#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-all}"
REF="${REF:-/opt/reference}"
OUT="${OUT:-$PWD/results}"
SOURCE="${SOURCE:-/opt/FlashRT-PAI/FlashRT-pi05-thor-limit-5421c93}"
export PYTHONPATH="$SOURCE"
export CUTE_DSL_ARCH=sm_101a
export LINGBOT_FA4_SRC="$SOURCE/csrc/attention/flash_attn_4_src"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "$OUT"
python doctor.py | tee "$OUT/runtime.json"
status=0
if [[ "$MODE" == pi05 || "$MODE" == all ]]; then
  PI05="${PI05:?Set PI05 to the converted OpenPI checkpoint}"
  FIXTURE="${FIXTURE:-$PWD/fixtures/libero_obs_2v_n8.npz}"
  export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PWD/openpi-cache}"
  mkdir -p "$OPENPI_DATA_HOME/big_vision" "$HOME/.cache/flash_rt"
  cp fixtures/paligemma_tokenizer.model "$OPENPI_DATA_HOME/big_vision/"
  cp fixtures/paligemma_tokenizer.model "$HOME/.cache/flash_rt/"
  "$REF/openpi-venv/bin/python" verify_pi05.py --mode openpi --checkpoint "$PI05" --fixture "$FIXTURE" --output "$OUT/openpi.npz" > "$OUT/openpi.log" 2>&1
  for tier in fp8 fp4; do
    python verify_pi05.py --mode "$tier" --checkpoint "$PI05" --fixture "$FIXTURE" --output "$OUT/pi05-$tier.npz" > "$OUT/pi05-$tier.log" 2>&1
    python compare_actions.py --reference "$OUT/openpi.npz" --candidate "$OUT/pi05-$tier.npz" --out "$OUT/pi05-$tier-accuracy.json" || status=1
  done
fi
if [[ "$MODE" == groot || "$MODE" == all ]]; then
  GROOT="${GROOT:?Set GROOT to the local-path GR00T overlay}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  GROOT_FIXTURE="${GROOT_FIXTURE:-$PWD/fixtures/groot-libero10-onecam.pt}"
  GROOT_EMBODIMENT="${GROOT_EMBODIMENT:-LIBERO_PANDA}"
  GROOT_NUM_VIEWS="${GROOT_NUM_VIEWS:-1}"
  GROOT_CPU_THREADS="${GROOT_CPU_THREADS:-2}"
  "$REF/groot-venv/bin/python" capture_groot.py --embodiment "$GROOT_EMBODIMENT" --flashrt "$SOURCE" --checkpoint "$GROOT" --input-fixture "$GROOT_FIXTURE" --out "$OUT/groot-reference.pt" > "$OUT/groot-capture.log" 2>&1
  for tier in fp8 fp4; do
    "$REF/groot-venv/bin/python" verify_groot_fixture.py --embodiment "$GROOT_EMBODIMENT" --num-views "$GROOT_NUM_VIEWS" --cpu-threads "$GROOT_CPU_THREADS" --checkpoint "$GROOT" --fixture "$GROOT_FIXTURE" --reference-records "$OUT/groot-reference.pt" --tier "$tier" --out "$OUT/groot-$tier.json" > "$OUT/groot-$tier.log" 2>&1 || status=1
  done
fi
echo "Results: $OUT; combined accuracy status: $status"
exit "$status"
