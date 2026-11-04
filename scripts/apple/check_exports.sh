#!/usr/bin/env bash
# Verify the complete defined external symbol set, including unexpected C++ ABI.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 LIBRARY EXPORTS_FILE" >&2
  exit 2
fi
library="$1"
allowlist="$2"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
LC_ALL=C sort "$allowlist" > "$scratch/expected"
xcrun nm -gU "$library" | awk 'NF {print $NF}' | LC_ALL=C sort > "$scratch/actual"
if ! diff -u "$scratch/expected" "$scratch/actual"; then
  echo "Unexpected export surface: $library (expected $allowlist)" >&2
  exit 1
fi
echo "Verified exports: $library"
