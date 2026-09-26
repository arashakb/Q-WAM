#!/usr/bin/env bash
# Prepare an ImageWAM checkout for Q-WAM:
#   1. apply patches/imagewam/*.patch (Q-WAM hooks in the RoboTwin policy; skipped if applied),
#   2. link the RoboTwin policy directory, as in the upstream README,
#   3. write $IMAGEWAM_ROOT/.env.local from the environment (an existing file is kept; FORCE=1
#      rewrites it). Unset paths default to the upstream README layout under $IMAGEWAM_ROOT.
#
#   IMAGEWAM_ROOT=<ImageWAM> [FLUX2_SRC=<flux2 source tree>] [FLUX2_MODEL_PATH=...]
#   [FLUX2_AE_MODEL_PATH=...] [CKPT_PATH=...] [DATASET_STATS_PATH=...] [PYTHON_BIN=...]
#   [HF_HOME=...] bash scripts/imagewam/setup_imagewam.sh
set -euo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${IMAGEWAM_ROOT:?set IMAGEWAM_ROOT to the ImageWAM checkout}"
IMAGEWAM_ROOT="$(cd "${IMAGEWAM_ROOT}" && pwd)"
UPSTREAM_COMMIT=00a4e7afe82a2245f77a95240730a572212edefd
cd "${IMAGEWAM_ROOT}"

head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "${head}" != "${UPSTREAM_COMMIT}" ]; then
  echo "[setup] WARNING: ImageWAM HEAD is ${head}; the patches were made against ${UPSTREAM_COMMIT}"
fi

for p in "${QWAM_ROOT}"/patches/imagewam/*.patch; do
  if git apply --reverse --check "${p}" 2>/dev/null; then
    echo "[setup] already applied: $(basename "${p}")"
  else
    git apply "${p}"
    echo "[setup] applied: $(basename "${p}")"
  fi
done

ln -sfn "${IMAGEWAM_ROOT}/experiments/robotwin/imagewam_policy" \
  "${IMAGEWAM_ROOT}/third_party/RoboTwin/policy/imagewam_policy"
echo "[setup] linked third_party/RoboTwin/policy/imagewam_policy"

ENV_FILE="${IMAGEWAM_ROOT}/.env.local"
if [ -f "${ENV_FILE}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[setup] keeping existing ${ENV_FILE} (FORCE=1 rewrites it)"
else
  CK="${IMAGEWAM_ROOT}/checkpoints"
  FLUX2_SRC="${FLUX2_SRC:-${IMAGEWAM_ROOT}/third_party/flux2}"
  FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${CK}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
  FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${CK}/flux2/FLUX.2-dev/ae.safetensors}"
  CKPT_PATH="${CKPT_PATH:-${CK}/imagewam_release/robotwin/flux2_klein_4b/model.pt}"
  DATASET_STATS_PATH="${DATASET_STATS_PATH:-${CK}/imagewam_release/robotwin/flux2_klein_4b/dataset_stats.json}"
  {
    echo "# ImageWAM RoboTwin evaluation paths (written by Q-WAM scripts/imagewam/setup_imagewam.sh)"
    printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN:-$(command -v python || true)}"
    printf 'FLUX2_SRC=%q\n' "${FLUX2_SRC}"
    printf 'FLUX2_MODEL_PATH=%q\n' "${FLUX2_MODEL_PATH}"
    printf 'FLUX2_AE_MODEL_PATH=%q\n' "${FLUX2_AE_MODEL_PATH}"
    printf 'FLUX2_QWEN3_MODEL_SPEC=%q\n' "${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
    printf 'CKPT_PATH=%q\n' "${CKPT_PATH}"
    printf 'DATASET_STATS_PATH=%q\n' "${DATASET_STATS_PATH}"
    if [ -n "${HF_HOME:-}" ]; then printf 'HF_HOME=%q\n' "${HF_HOME}"; fi
  } > "${ENV_FILE}"
  echo "[setup] wrote ${ENV_FILE}"
  for f in "${FLUX2_MODEL_PATH}" "${FLUX2_AE_MODEL_PATH}" "${CKPT_PATH}" "${DATASET_STATS_PATH}"; do
    [ -e "${f}" ] || echo "[setup] WARNING: missing ${f}"
  done
  [ -d "${FLUX2_SRC}" ] || echo "[setup] WARNING: missing FLUX.2 source tree ${FLUX2_SRC}"
fi
echo "[setup] done"
