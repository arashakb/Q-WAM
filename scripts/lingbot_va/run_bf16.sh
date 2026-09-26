#!/usr/bin/env bash
# bf16 LingBot-VA reference on RoboTwin 2.0, clean and randomized, then the success rates.
#
#   LINGBOT_ROOT     patched lingbot-va checkout (setup_lingbot.sh)
#   NGPU, GPU_IDS, START_PORT, MASTER_PORT, PYTHON, ROBOTWIN_ROOT   passed to run_8gpu_queue.sh
#
# Rerunning resumes: finished tasks are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LINGBOT_ROOT:?set LINGBOT_ROOT to the patched lingbot-va checkout}"
export LINGBOT_ROOT
PYTHON="${PYTHON:-python}"
TEST_NUM="${TEST_NUM:-50}"
unset LB_QUANT_CKPT
for cond in clean randomized; do
  ARM_TAG="lingbot_bf16_${cond}" TASK_CONFIG="demo_${cond}" TEST_NUM="$TEST_NUM" \
    bash "$HERE/run_8gpu_queue.sh"
done
"$PYTHON" "$HERE/read_results.py" -n "$TEST_NUM" lingbot_bf16_clean lingbot_bf16_randomized
