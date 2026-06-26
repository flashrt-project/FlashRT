#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-candidate}"
SHAPE="${2:-gate}"
NCU="${NCU:-/usr/local/cuda-12.4/bin/ncu}"

"${SCRIPT_DIR}/build.sh" >/dev/null
mkdir -p "${SCRIPT_DIR}/profiles"

REP="${SCRIPT_DIR}/profiles/${MODE}_${SHAPE}"
CSV="${SCRIPT_DIR}/profiles/${MODE}_${SHAPE}_details.csv"

"${NCU}" \
  --force-overwrite \
  --set full \
  --target-processes all \
  --kernel-name 'regex:.*fp8_bs_gemm_kernel.*' \
  --launch-skip 5 \
  --launch-count 1 \
  -o "${REP}" \
  "${SCRIPT_DIR}/build/bench_sm89_fp8_block128_gemm" \
    --shape "${SHAPE}" \
    --mode "${MODE}" \
    --warmup 5 \
    --iters 1 \
    --flush-l2-mb 256

"${NCU}" --import "${REP}.ncu-rep" --page details --csv > "${CSV}"
echo "${CSV}"
