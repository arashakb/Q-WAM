#!/usr/bin/env bash
# Q-WAM on Fast-WAM, evaluated on RoboTwin 2.0 (paper configuration by default).
#
#   bash scripts/fastwam/run_qwam.sh <clean|randomized> [task|all] [episodes=100]
#
# W4A4 with weights and activations quantized in groups of 32, SmoothQuant smoothing (alpha 0.5),
# block Hadamard rotation, and ASP with rank 32 on the 300 action-expert linears. Inputs:
#   QWAM_ABSMAX  activation absmax from merge_absmax.py   (default: $QWAM_WORK/absmax.pt)
#   QWAM_AOG     action-expert AOGs from merge_aog.py     (default: $QWAM_WORK/aog_action.pt)
#   QWAM_WORK    default: <this repository>/work/fastwam
# Component ablation (paper Table 3); each setting gets its own default TAG:
#   QWAM_RANK=0                                "+ smoothing and rotation"
#   QWAM_RANK=0 QWAM_ROTATE=0 QWAM_SMOOTH=0    "per-group W4A4"
# FASTWAM_ROOT, GPUS, TAG and the other variables are those of run_robotwin_eval.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWAM_ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${1:?usage: run_qwam.sh <clean|randomized> [task|all] [episodes]}"
QWAM_WORK="${QWAM_WORK:-$QWAM_ROOT/work/fastwam}"

export QWAM_ENABLE=1
export QWAM_RANK="${QWAM_RANK:-32}" QWAM_ROTATE="${QWAM_ROTATE:-1}" QWAM_SMOOTH="${QWAM_SMOOTH:-1}"
QWAM_ABSMAX="${QWAM_ABSMAX:-$QWAM_WORK/absmax.pt}"
QWAM_AOG="${QWAM_AOG:-$QWAM_WORK/aog_action.pt}"
# The policy runs from the RoboTwin directory, so the caches are passed as absolute paths.
if [ "$QWAM_SMOOTH" != "0" ]; then
  [ -f "$QWAM_ABSMAX" ] || { echo "missing activation absmax: $QWAM_ABSMAX" >&2; exit 1; }
  export QWAM_ABSMAX="$(readlink -f "$QWAM_ABSMAX")"
fi
if [ "$QWAM_RANK" != "0" ]; then
  [ -f "$QWAM_AOG" ] || { echo "missing action AOG: $QWAM_AOG" >&2; exit 1; }
  export QWAM_AOG="$(readlink -f "$QWAM_AOG")"
fi

if [ "$QWAM_RANK" = "32" ] && [ "$QWAM_ROTATE" = "1" ] && [ "$QWAM_SMOOTH" = "1" ]; then
  ARM=qwam
else
  ARM="qwam_r${QWAM_RANK}"
  [ "$QWAM_ROTATE" = "1" ] || ARM+="_norot"
  [ "$QWAM_SMOOTH" = "1" ] || ARM+="_nosmooth"
fi
export TAG="${TAG:-${ARM}_${MODE}}"
exec bash "$HERE/run_robotwin_eval.sh" "$MODE" "${2:-all}" "${3:-100}"
