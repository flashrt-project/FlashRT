#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv-onnx-modelopt}"
INSTALL_MODELOPT="${INSTALL_MODELOPT:-1}"
MODELOPT_EXTRAS="${MODELOPT_EXTRAS:-onnx}"
SYSTEM_SITE_PACKAGES="${SYSTEM_SITE_PACKAGES:-0}"

if [[ "${SYSTEM_SITE_PACKAGES}" == "1" ]]; then
  python3 -m venv --system-site-packages "${ENV_DIR}"
else
  python3 -m venv "${ENV_DIR}"
fi
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install onnx onnxruntime numpy

if [[ "${INSTALL_MODELOPT}" == "1" ]]; then
  if [[ "${SYSTEM_SITE_PACKAGES}" == "1" ]]; then
    "${ENV_DIR}/bin/python" -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      --upgrade --no-deps nvidia-modelopt
    "${ENV_DIR}/bin/python" -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      onnxruntime onnx-graphsurgeon onnxslim onnxconverter-common \
      onnxscript polygraphy lief
  else
    "${ENV_DIR}/bin/python" -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      "nvidia-modelopt[${MODELOPT_EXTRAS}]" \
      onnx-graphsurgeon
  fi
fi

cat <<EOF
Created ${ENV_DIR}

Activate with:
  source ${ENV_DIR}/bin/activate

Set INSTALL_MODELOPT=0 to install only ONNX export dependencies.
Set MODELOPT_EXTRAS=all to install the full ModelOpt extra set.
Set SYSTEM_SITE_PACKAGES=1 inside a PyTorch container to reuse its torch install.
EOF
