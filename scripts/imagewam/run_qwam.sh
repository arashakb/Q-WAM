#!/usr/bin/env bash
# Q-WAM on ImageWAM, paper configuration (Table 1): export the W4A4 checkpoint (weight and
# activation group 32, alpha 0.5, block Hadamard with block cap 1024, ASP rank 32 on the action
# expert with the shipped subspace file), then evaluate it on RoboTwin 2.0, clean and randomized,
# 50 tasks x 100 episodes each (two runs).
#
#   IMAGEWAM_ROOT=<ImageWAM> [GPU_IDS=0,...,7] bash scripts/imagewam/run_qwam.sh
#
#   IW_QUANT_CKPT   checkpoint to evaluate; exported first when it does not exist
#                   (default work/imagewam/imagewam_w4a4_qwam_r32_g32.pt)
#   IW_ABSMAX       calibration absmax for the export (default: the shipped paper calibration)
#   EXPORT_ARGS     extra export_qwam.py flags, e.g. the Table 3 rows:
#                     "--subspaces none"                          smoothing and rotation
#                     "--no-smooth --no-rotate --subspaces none"  per-group W4A4
#   ARM_PREFIX=qwam run label prefix (<prefix>_clean, <prefix>_random)
set -eo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${IMAGEWAM_ROOT:?set IMAGEWAM_ROOT to the prepared ImageWAM checkout}"
if [ -f "${IMAGEWAM_ROOT}/.env.local" ]; then
  set -a; source "${IMAGEWAM_ROOT}/.env.local"; set +a
fi

CKPT="${IW_QUANT_CKPT:-${QWAM_ROOT}/work/imagewam/imagewam_w4a4_qwam_r32_g32.pt}"
if [ ! -f "${CKPT}" ]; then
  # shellcheck disable=SC2086
  PYTHONPATH="${QWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN:-python}" \
    "${QWAM_ROOT}/scripts/imagewam/export_qwam.py" \
    --base-ckpt "${IW_BASE_CKPT:-${CKPT_PATH:-${IMAGEWAM_ROOT}/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt}}" \
    ${IW_ABSMAX:+--absmax "${IW_ABSMAX}"} --out "${CKPT}" ${EXPORT_ARGS:-}
fi

for PHASE in clean random; do
  ARM="${ARM_PREFIX:-qwam}_${PHASE}" PHASES="[${PHASE}]" TASKS="" EPISODES=100 \
    IW_CACHE_WEIGHTS=1 IW_QUANT_CKPT="${CKPT}" \
    bash "${QWAM_ROOT}/scripts/imagewam/run_imagewam_robotwin.sh"
done
