#!/usr/bin/env bash
# bf16 Fast-WAM (the upper-bound rows of the paper) on RoboTwin 2.0.
#
#   bash scripts/fastwam/run_bf16.sh <clean|randomized> [task|all] [episodes=100]
#
# FASTWAM_ROOT, GPUS, TAG (default bf16_<mode>) and the other variables are those of run_robotwin_eval.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?usage: run_bf16.sh <clean|randomized> [task|all] [episodes]}"
export QWAM_ENABLE=0
export TAG="${TAG:-bf16_${MODE}}"
exec bash "$HERE/run_robotwin_eval.sh" "$MODE" "${2:-all}" "${3:-100}"
