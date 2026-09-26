#!/usr/bin/env bash
# Smoothing calibration: one bf16 closed-loop clean episode per RoboTwin task (50 tasks) with
# per-input-channel absmax hooks on the MoT Linears, then the elementwise-max merge of the per-task
# shards over the 154 Linears that execute.
#
#   IMAGEWAM_ROOT=<ImageWAM> CALIB_DIR=<empty dir> [GPU_IDS=0,...,7 NUM_GPUS=8] \
#     bash scripts/imagewam/calibrate_absmax.sh
#   -> ${CALIB_DIR}/imagewam_act_absmax_c50.pt
set -eo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${IMAGEWAM_ROOT:?set IMAGEWAM_ROOT to the prepared ImageWAM checkout}"
: "${CALIB_DIR:?set CALIB_DIR to an empty output directory}"
mkdir -p "${CALIB_DIR}"
CALIB_DIR="$(cd "${CALIB_DIR}" && pwd)"
if compgen -G "${CALIB_DIR}/absmax_*.pt" > /dev/null; then
  echo "${CALIB_DIR} already holds absmax shards; use an empty directory" >&2
  exit 2
fi
if [ -f "${IMAGEWAM_ROOT}/.env.local" ]; then
  set -a; source "${IMAGEWAM_ROOT}/.env.local"; set +a
fi

ARM=bf16_calib TASKS="" EPISODES=1 PHASES="[clean]" IW_CALIB_OUT="${CALIB_DIR}/absmax" \
  bash "${QWAM_ROOT}/scripts/imagewam/run_imagewam_robotwin.sh"

PYTHONPATH="${QWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN:-python}" \
  "${QWAM_ROOT}/scripts/imagewam/merge_absmax.py" --dir "${CALIB_DIR}" --prefix absmax \
  --out "${CALIB_DIR}/imagewam_act_absmax_c50.pt" --expect-layers 154
