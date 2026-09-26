#!/usr/bin/env bash
# bf16 reference of ImageWAM on RoboTwin 2.0, clean and randomized, 50 tasks x 100 episodes each.
#
#   IMAGEWAM_ROOT=<ImageWAM> [GPU_IDS=0,...,7] bash scripts/imagewam/run_bf16.sh
set -eo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${IMAGEWAM_ROOT:?set IMAGEWAM_ROOT to the prepared ImageWAM checkout}"

for PHASE in clean random; do
  ARM="bf16_${PHASE}" PHASES="[${PHASE}]" TASKS="" EPISODES=100 \
    bash "${QWAM_ROOT}/scripts/imagewam/run_imagewam_robotwin.sh"
done
