#!/usr/bin/env bash
# Evaluate Fast-WAM on RoboTwin 2.0 with FastWAM's single-task entry point, one task per GPU at a time.
#
#   bash scripts/fastwam/run_robotwin_eval.sh <clean|randomized> <task|all> [episodes=100]
#
# Environment:
#   FASTWAM_ROOT  FastWAM checkout prepared with scripts/fastwam/setup_fastwam.sh          (required)
#   TAG           run name (default: eval_<mode>); results are written to
#                   $FASTWAM_ROOT/evaluate_results/robotwin/<checkpoint name>/<TAG>/<task>/_result_<clean|random>.txt
#                 and per-task logs to .../<TAG>/logs/<task>.log
#   GPUS          GPU ids, one worker each (default: all GPUs listed by nvidia-smi)
#   CKPT, STATS   checkpoint and dataset statistics (default: the released RoboTwin checkpoint)
#   PYTHON        python of the FastWAM environment (default: python)
#   FRESH=1       also re-run tasks that already have a complete result (default: they are skipped)
# Q-WAM is switched on by the QWAM_* variables, which the policy inherits (see run_qwam.sh).
set -euo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${FASTWAM_ROOT:?set FASTWAM_ROOT to the FastWAM checkout}"
FASTWAM_ROOT="$(cd "$FASTWAM_ROOT" && pwd)"

usage() { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 1; }
MODE="${1:-}"; TASK="${2:-}"; EPISODES="${3:-100}"
[ -n "$MODE" ] && [ -n "$TASK" ] || usage
case "$MODE" in
  clean)      TASK_CONFIG=demo_clean ;;
  randomized) TASK_CONFIG=demo_randomized ;;
  *) echo "first argument must be clean or randomized (got '$MODE')" >&2; usage ;;
esac

PYTHON="${PYTHON:-python}"
CKPT="${CKPT:-$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
STATS="${STATS:-$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
TAG="${TAG:-eval_$MODE}"
FRESH="${FRESH:-0}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)}"
read -r -a GPU_IDS <<< "${GPUS:-0}"
TASK_LIST="$FASTWAM_ROOT/third_party/RoboTwin/task_config/_eval_step_limit.yml"
for f in "$CKPT" "$STATS" "$TASK_LIST"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done

# The entry point names the run directory after the checkpoint file and EVALUATION.output_dir.
RUN_DIR="$FASTWAM_ROOT/evaluate_results/robotwin/$(basename "$CKPT" .pt)/$TAG"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$FASTWAM_ROOT"
export PYTHONPATH="$QWAM_ROOT:$FASTWAM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$FASTWAM_ROOT/checkpoints}"
if [ "${QWAM_ENABLE:-0}" = "1" ]; then
  grep -q "QWAM_ENABLE" experiments/robotwin/fastwam_policy/deploy_policy.py || {
    echo "the RoboTwin policy has no Q-WAM hook; run scripts/fastwam/setup_fastwam.sh first" >&2; exit 1; }
  "$PYTHON" -c "import qwam.fastwam" || { echo "cannot import qwam.fastwam with $PYTHON" >&2; exit 1; }
fi

run_task() {  # task gpu
  "$PYTHON" experiments/robotwin/eval_robotwin_single.py \
    ckpt="$CKPT" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.task_name="$1" \
    EVALUATION.task_config="$TASK_CONFIG" \
    EVALUATION.eval_num_episodes="$EPISODES" \
    EVALUATION.output_dir="./evaluate_results/robotwin/$TAG" \
    gpu_id="$2"
}

is_done() {  # task -> success only if its log shows a finished run of $EPISODES episodes
  local log="$LOG_DIR/$1.log" text n
  [ "$FRESH" = "1" ] && return 1
  [ -f "$log" ] || return 1
  text="$(sed 's/\x1b\[[0-9;]*m//g' "$log")"
  grep -q "Data has been saved to" <<< "$text" || return 1
  n="$(grep -oE '/[0-9]+ =>' <<< "$text" | tr -cd '0-9\n' | sort -n | tail -1)"
  [ -n "$n" ] && [ "$n" -ge "$EPISODES" ]
}

echo ">>> $MODE ($TASK_CONFIG) | tag $TAG | $EPISODES episodes | QWAM_ENABLE=${QWAM_ENABLE:-0}"
if [ "$TASK" != "all" ]; then
  run_task "$TASK" "${GPU_IDS[0]}" 2>&1 | tee "$LOG_DIR/$TASK.log"
  exit "${PIPESTATUS[0]}"
fi

# Longest tasks first to shorten the total wall time; the order does not affect any result
# (each task has its own seeds). Tasks missing from this list run last.
ORDER="open_microwave put_bottles_dustbin put_object_cabinet hanging_mug place_can_basket
stack_bowls_three blocks_ranking_size place_object_basket stack_blocks_three blocks_ranking_rgb
handover_block place_cans_plasticbox handover_mic place_dual_shoes scan_object stack_bowls_two
place_burger_fries stack_blocks_two place_bread_basket dump_bin_bigbin move_can_pot
pick_diverse_bottles place_bread_skillet move_stapler_pad move_pillbottle_pad place_fan turn_switch
pick_dual_bottles stamp_seal lift_pot rotate_qrcode place_mouse_pad open_laptop adjust_bottle
place_a2b_left place_a2b_right place_shoe place_object_scale place_empty_cup place_object_stand
place_phone_stand place_container_plate beat_block_hammer shake_bottle shake_bottle_horizontally
move_playingcard_away press_stapler grab_roller click_alarmclock click_bell"
mapfile -t ALL < <(grep -E '^[a-z0-9_]+:' "$TASK_LIST" | sed 's/:.*//')
declare -A KNOWN=() QUEUED=()
for t in "${ALL[@]}"; do KNOWN[$t]=1; done
TASKS=()
for t in $ORDER; do if [ -n "${KNOWN[$t]:-}" ]; then TASKS+=("$t"); QUEUED[$t]=1; fi; done
for t in "${ALL[@]}"; do if [ -z "${QUEUED[$t]:-}" ]; then TASKS+=("$t"); fi; done
[ "${#TASKS[@]}" -eq "${#ALL[@]}" ] || { echo "task list mismatch" >&2; exit 1; }

# Dynamic queue: each GPU worker takes the next task index under a lock.
QUEUE="$RUN_DIR/.queue"
rm -rf "$QUEUE"; mkdir -p "$QUEUE"; echo 0 > "$QUEUE/idx"
worker() {  # gpu
  local gpu="$1" i t
  while true; do
    exec 9>"$QUEUE/lock"; flock 9
    i="$(cat "$QUEUE/idx")"; echo $((i + 1)) > "$QUEUE/idx"
    flock -u 9; exec 9>&-
    [ "$i" -ge "${#TASKS[@]}" ] && break
    t="${TASKS[$i]}"
    if is_done "$t"; then echo "[gpu$gpu] skip $t (complete result)"; continue; fi
    echo "[gpu$gpu] ($((i + 1))/${#TASKS[@]}) $t"
    if run_task "$t" "$gpu" > "$LOG_DIR/$t.log" 2>&1; then
      echo "[gpu$gpu] done $t"
    else
      echo "[gpu$gpu] FAILED $t (see $LOG_DIR/$t.log)"
    fi
  done
}
echo ">>> ${#TASKS[@]} tasks on GPUs [${GPU_IDS[*]}]"
for g in "${GPU_IDS[@]}"; do worker "$g" & done
wait
rm -rf "$QUEUE"
"$PYTHON" "$QWAM_ROOT/scripts/fastwam/success_rate.py" "$RUN_DIR"
