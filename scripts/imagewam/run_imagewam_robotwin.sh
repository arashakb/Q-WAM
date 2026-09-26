#!/usr/bin/env bash
# RoboTwin 2.0 evaluation of ImageWAM (FLUX.2 klein 4B) through the upstream manager, bf16 or W4A4.
#
#   IMAGEWAM_ROOT       ImageWAM checkout prepared with scripts/imagewam/setup_imagewam.sh
#   TASKS=""            all 50 tasks; "a,b" for a subset
#   EPISODES=100        episodes per task (paper protocol)
#   PHASES="[clean]"    "[clean]", "[random]" or "[clean,random]"
#   NUM_GPUS=8  MAX_TASKS_PER_GPU=1  GPU_IDS="0,1,..."   (GPU_IDS pins physical GPU ids)
#   ARM=bf16            run label; bf16 and bf16_* run unquantized, any other label needs IW_QUANT_CKPT
#   IW_QUANT_CKPT       checkpoint from scripts/imagewam/export_qwam.py
#   IW_CACHE_WEIGHTS=1  keep dequantized weights resident while GPU memory allows (same outputs)
#   IW_CALIB_OUT        write activation absmax shards with this path prefix (calibration only)
#   IW_EXTRA_ARGS       extra Hydra overrides, e.g. EVALUATION.output_dir=<run dir> to resume a run
#
# Results: $IMAGEWAM_ROOT/evaluate_results/robotwin/<ckpt tag>/<timestamp>/ (read_results.py).
set -eo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${IMAGEWAM_ROOT:?set IMAGEWAM_ROOT to the prepared ImageWAM checkout}"

# Workers run from IMAGEWAM_ROOT, so user paths are made absolute first.
abspath() { if [ -d "$(dirname "$1")" ]; then echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; else echo "$1"; fi; }
if [ -n "${IW_QUANT_CKPT:-}" ]; then IW_QUANT_CKPT="$(abspath "${IW_QUANT_CKPT}")"; fi
if [ -n "${IW_CALIB_OUT:-}" ]; then
  mkdir -p "$(dirname "${IW_CALIB_OUT}")"
  IW_CALIB_OUT="$(abspath "${IW_CALIB_OUT}")"
fi
cd "${IMAGEWAM_ROOT}"

# qwam must be importable by the per-task workers the manager spawns.
export PYTHONPATH="${QWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Keep ~/.local site-packages out of the workers.
export PYTHONNOUSERSITE=1

TASKS="${TASKS:-}"
EPISODES="${EPISODES:-100}"
PHASES="${PHASES:-[clean]}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
ARM="${ARM:-bf16}"

case "$ARM" in
  bf16|bf16_*)
    unset IW_QUANT_CKPT
    ;;
  *)
    : "${IW_QUANT_CKPT:?ARM=$ARM requires IW_QUANT_CKPT}"
    [ -f "${IW_QUANT_CKPT}" ] || { echo "no checkpoint at ${IW_QUANT_CKPT}" >&2; exit 2; }
    export IW_QUANT_CKPT
    ;;
esac
export IMAGEWAM_ARM="$ARM"

# Runtime variables must reach the workers.
for v in IW_CACHE_WEIGHTS IW_CALIB_OUT; do
  if [ -n "${!v:-}" ]; then
    export "$v"
    echo "[imagewam-eval] $v=${!v}"
  fi
done

ARGS=(
  EVALUATION.eval_num_episodes="${EPISODES}"
  MULTIRUN.phases="${PHASES}"
)
if [ -n "${GPU_IDS:-}" ]; then
  ARGS+=("MULTIRUN.gpu_ids=[${GPU_IDS}]")
fi
if [ -n "${IW_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${IW_EXTRA_ARGS})
fi
if [ -n "$TASKS" ]; then
  # quoted so Hydra receives one string (the manager splits on commas)
  ARGS+=("EVALUATION.task_name='${TASKS}'")
fi

echo "[imagewam-eval] arm=${ARM} gpus=${NUM_GPUS} episodes=${EPISODES} phases=${PHASES} tasks=${TASKS:-<all 50>}"
if [ -n "${IW_QUANT_CKPT:-}" ]; then
  echo "[imagewam-eval] quant ckpt=${IW_QUANT_CKPT}"
fi

# EVAL_NUM_EPISODES also sets the upstream default, so both episode overrides agree.
exec env NUM_GPUS="${NUM_GPUS}" MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU}" FLUX2_VARIANT=4b \
  EVAL_NUM_EPISODES="${EPISODES}" \
  bash scripts/flux2/run_eval_flux2_robotwin.sh "${ARGS[@]}"
