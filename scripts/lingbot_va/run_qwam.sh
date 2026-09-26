#!/usr/bin/env bash
# Q-WAM on LingBot-VA: export the W4A4 checkpoint (unless it exists), evaluate it on RoboTwin 2.0
# under the clean and the randomized condition, and print the success rates.
#
#   LINGBOT_ROOT     patched lingbot-va checkout (setup_lingbot.sh)
#   VARIANT=qwam     qwam      smoothing + rotation + ASP rank 32 (Table 1)
#                    smoothrot smoothing + rotation, no ASP        (Table 3)
#                    pergroup  per-group W4A4 only                 (Table 3)
#   SUBSPACES        ASP subspace file for VARIANT=qwam
#                    (default work/lingbot_va/lingbot_va_asp_subspaces_r32.pt, build_asp_subspaces.py)
#   ABSMAX           activation absmax (default artifacts/lingbot_va/lingbot_va_act_absmax.pt)
#   CKPT             checkpoint path (default work/lingbot_va/lingbot_va_w4a4_<VARIANT>.pt)
#   NGPU, GPU_IDS, START_PORT, MASTER_PORT, PYTHON, ROBOTWIN_ROOT   passed to run_8gpu_queue.sh
#
# Rerunning resumes: finished tasks are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWAM_ROOT="$(cd "$HERE/../.." && pwd)"
: "${LINGBOT_ROOT:?set LINGBOT_ROOT to the patched lingbot-va checkout}"
export LINGBOT_ROOT
export PYTHONPATH="$QWAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
WORK="${WORK:-$QWAM_ROOT/work/lingbot_va}"
VARIANT="${VARIANT:-qwam}"
TEST_NUM="${TEST_NUM:-50}"
ABSMAX="${ABSMAX:-$QWAM_ROOT/artifacts/lingbot_va/lingbot_va_act_absmax.pt}"
SUBSPACES="${SUBSPACES:-$WORK/lingbot_va_asp_subspaces_r32.pt}"
case "$VARIANT" in
  qwam)      EXPORT_ARGS=(--subspaces "$SUBSPACES") ;;
  smoothrot) EXPORT_ARGS=(--subspaces none) ;;
  pergroup)  EXPORT_ARGS=(--subspaces none --no-smooth --no-rotate) ;;
  *) echo "VARIANT must be qwam, smoothrot or pergroup" >&2; exit 1 ;;
esac
CKPT="${CKPT:-$WORK/lingbot_va_w4a4_${VARIANT}.pt}"

if [ ! -s "$CKPT" ]; then
  "$PYTHON" "$HERE/export_qwam.py" --absmax "$ABSMAX" "${EXPORT_ARGS[@]}" --out "$CKPT"
fi
for cond in clean randomized; do
  ARM_TAG="lingbot_${VARIANT}_${cond}" TASK_CONFIG="demo_${cond}" TEST_NUM="$TEST_NUM" \
    LB_QUANT_CKPT="$CKPT" bash "$HERE/run_8gpu_queue.sh"
done
"$PYTHON" "$HERE/read_results.py" -n "$TEST_NUM" "lingbot_${VARIANT}_clean" "lingbot_${VARIANT}_randomized"
