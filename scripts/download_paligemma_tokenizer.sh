#!/usr/bin/env bash
# Download the PaliGemma SentencePiece tokenizer required by Pi0 /
# Pi0.5. The file lives in Google's public big_vision bucket; no
# auth, no Cloud SDK, no Python deps.
#
# Default cache: ~/.cache/flash_rt/paligemma_tokenizer.model
# Override:      $FLASH_RT_PALIGEMMA_TOKENIZER (set to a target path)
#
# Usage:
#     bash scripts/download_paligemma_tokenizer.sh
#     # or, custom location:
#     FLASH_RT_PALIGEMMA_TOKENIZER=/data/paligemma.model \
#         bash scripts/download_paligemma_tokenizer.sh

set -euo pipefail

URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
DEST="${FLASH_RT_PALIGEMMA_TOKENIZER:-$HOME/.cache/flash_rt/paligemma_tokenizer.model}"
EXPECTED_MD5="1420adc9856720a559e8a87284b195e2"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
    echo "Already present: $DEST"
    if command -v md5sum >/dev/null; then
        actual_md5=$(md5sum "$DEST" | awk '{print $1}')
        if [[ "$actual_md5" == "$EXPECTED_MD5" ]]; then
            echo "MD5 matches; nothing to do."
            exit 0
        fi
        echo "MD5 mismatch (got $actual_md5, expected $EXPECTED_MD5);"
        echo "re-downloading..."
    else
        echo "(md5sum not available — skipping integrity check.)"
        exit 0
    fi
fi

echo "Downloading paligemma_tokenizer.model (~4.1 MiB) ..."
echo "  from: $URL"
echo "  to:   $DEST"

if command -v curl >/dev/null; then
    curl -fSL "$URL" -o "$DEST"
elif command -v wget >/dev/null; then
    wget -q "$URL" -O "$DEST"
else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
fi

if command -v md5sum >/dev/null; then
    actual_md5=$(md5sum "$DEST" | awk '{print $1}')
    if [[ "$actual_md5" != "$EXPECTED_MD5" ]]; then
        echo "ERROR: MD5 mismatch after download." >&2
        echo "  expected: $EXPECTED_MD5" >&2
        echo "  got:      $actual_md5" >&2
        exit 2
    fi
    echo "MD5 OK ($actual_md5)."
fi

bytes=$(stat -c '%s' "$DEST" 2>/dev/null || stat -f '%z' "$DEST")
echo "Done. ($bytes bytes)"
