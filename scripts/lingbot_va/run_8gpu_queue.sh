#!/usr/bin/env bash
# RoboTwin 2.0 evaluation of LingBot-VA: one inference server per GPU and a work queue that hands the
# next task to whichever GPU becomes free.
#
#   LINGBOT_ROOT       patched lingbot-va checkout (scripts/lingbot_va/setup_lingbot.sh)
#   ROBOTWIN_ROOT      RoboTwin checkout (default $LINGBOT_ROOT/RoboTwin)
#   ARM_TAG            run label; episodes go to $ROBOTWIN_ROOT/results_<ARM_TAG>/stseed-10000/
#                      visualization/<task>/<idx>_<instruction>_<True|False>.mp4. A tag containing
#                      "bf16" runs the unquantized model; any other tag requires LB_QUANT_CKPT.
#   LB_QUANT_CKPT      checkpoint written by export_qwam.py
#   TASK_CONFIG        demo_clean | demo_randomized
#   TEST_NUM           episodes per task (default 50)
#   NGPU=8, GPU_IDS    GPUs to use (default 0..NGPU-1); one server and one client per GPU
#   START_PORT=29556   server i listens on START_PORT+i; its torchrun store uses MASTER_PORT+i
#   MASTER_PORT=29661
#   PYTHON=python      interpreter of the environment that runs LingBot-VA and RoboTwin
#   LOG_DIR            default <Q-WAM>/logs/lingbot_va
#   RESUME_DRYRUN=1    only report which tasks would run
#
# Resume: a task with >= TEST_NUM recorded episodes is skipped; a partially recorded task is moved to
# results_<ARM_TAG>/_archived_partial/ and rerun from episode 0 (the client always starts there).
# Every server and client started here carries QWAM_RUN_ID=<id of this run> in its environment; on
# exit (normal, error, SIGINT or SIGTERM) exactly those processes are stopped, and nothing else is
# ever signalled.
set -uo pipefail

QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${LINGBOT_ROOT:?set LINGBOT_ROOT to the patched lingbot-va checkout}"
: "${ARM_TAG:?set ARM_TAG}"
LINGBOT_ROOT="$(cd "$LINGBOT_ROOT" && pwd)"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$LINGBOT_ROOT/RoboTwin}"
[ -d "$ROBOTWIN_ROOT" ] || { echo "no RoboTwin checkout at $ROBOTWIN_ROOT" >&2; exit 1; }
ROBOTWIN_ROOT="$(cd "$ROBOTWIN_ROOT" && pwd)"
export PYTHONPATH="$QWAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TEST_NUM="${TEST_NUM:-50}"
NGPU="${NGPU:-8}"
START_PORT="${START_PORT:-29556}"
MASTER_PORT="${MASTER_PORT:-29661}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"
LOG_DIR="${LOG_DIR:-$QWAM_ROOT/logs/lingbot_va}"
RES="$ROBOTWIN_ROOT/results_${ARM_TAG}"

case "$TASK_CONFIG" in
  demo_clean|demo_randomized) ;;
  *) echo "TASK_CONFIG must be demo_clean or demo_randomized, got $TASK_CONFIG" >&2; exit 1 ;;
esac
IFS=',' read -r -a GPUS <<< "${GPU_IDS:-$(seq -s, 0 $((NGPU - 1)))}"
if [ "${#GPUS[@]}" -ne "$NGPU" ]; then
  echo "GPU_IDS has ${#GPUS[@]} entries but NGPU=$NGPU" >&2; exit 1
fi

# The arm tag declares what is evaluated; refuse a quantized tag without a checkpoint and vice versa.
case "$ARM_TAG" in
  *bf16*)
    if [ -n "${LB_QUANT_CKPT:-}" ]; then
      echo "ARM_TAG=$ARM_TAG names a bf16 run but LB_QUANT_CKPT is set" >&2; exit 1
    fi
    unset LB_QUANT_CKPT ;;
  *)
    if [ -z "${LB_QUANT_CKPT:-}" ]; then
      echo "ARM_TAG=$ARM_TAG needs LB_QUANT_CKPT (use a tag containing 'bf16' for the bf16 model)" >&2
      exit 1
    fi
    [ -s "$LB_QUANT_CKPT" ] || { echo "no checkpoint at $LB_QUANT_CKPT" >&2; exit 1; }
    LB_QUANT_CKPT="$(cd "$(dirname "$LB_QUANT_CKPT")" && pwd)/$(basename "$LB_QUANT_CKPT")"
    export LB_QUANT_CKPT ;;
esac

mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/queue_${ARM_TAG}.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"; }

TASKS=(
  stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet
  stack_blocks_three place_shoe adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad
  pick_dual_bottles shake_bottle place_fan turn_switch shake_bottle_horizontally
  place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad
  place_burger_fries place_bread_basket pick_diverse_bottles open_microwave beat_block_hammer
  press_stapler click_bell move_playingcard_away open_laptop move_can_pot stack_bowls_two
  place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two
  place_cans_plasticbox click_alarmclock blocks_ranking_size place_phone_stand place_can_basket
  place_object_scale place_a2b_left grab_roller place_dual_shoes place_empty_cup blocks_ranking_rgb
)
log "========= $ARM_TAG: ${#TASKS[@]} tasks, test_num=$TEST_NUM, task_config=$TASK_CONFIG ========="
log "checkpoint: ${LB_QUANT_CKPT:-<none: bf16>}"
log "results:    $RES"

# ---- resume ----------------------------------------------------------------------------------
episodes_done() {   # $1 = task -> number of distinct episode indices recorded
  local t=$1 d n=0 c
  for d in "$RES"/stseed-*/visualization/"$t"; do
    [ -d "$d" ] || continue
    c=$(ls "$d" 2>/dev/null | sed -nE 's/^([0-9]+)_.*_(True|False)\.mp4$/\1/p' | sort -u | wc -l)
    n=$((n + c))
  done
  echo "$n"
}
KEEP=(); n_skip=0; n_restart=0
for t in "${TASKS[@]}"; do
  n=$(episodes_done "$t")
  if [ "$n" -ge "$TEST_NUM" ]; then
    n_skip=$((n_skip + 1))
  else
    if [ "$n" -gt 0 ]; then
      if [ -z "${RESUME_DRYRUN:-}" ]; then
        mkdir -p "$RES/_archived_partial"
        for d in "$RES"/stseed-*/visualization/"$t"; do
          [ -d "$d" ] && mv "$d" "$RES/_archived_partial/${t}_$(date +%Y%m%d_%H%M%S)"
        done
      fi
      log "  resume: restart $t ($n < $TEST_NUM episodes recorded, partial archived)"
      n_restart=$((n_restart + 1))
    fi
    KEEP+=("$t")
  fi
done
log "resume: $n_skip complete, $n_restart partial restarted, ${#KEEP[@]} to run"
if [ "${#KEEP[@]}" -eq 0 ]; then log "nothing to run"; exit 0; fi
TASKS=("${KEEP[@]}")
if [ -n "${RESUME_DRYRUN:-}" ]; then log "RESUME_DRYRUN set: stopping before any GPU is used"; exit 0; fi

# ---- processes of this run ---------------------------------------------------------------------
RUN_ID="lingbot-$$-$RANDOM$RANDOM"
ours() {            # PIDs of live processes whose environment carries this run's marker
  grep -lzxF "QWAM_RUN_ID=$RUN_ID" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3
}
stop_ours() {
  local pids k
  pids=$(ours)
  [ -n "$pids" ] || return 0
  log "stopping $(echo "$pids" | wc -w) processes started by this run"
  kill -TERM $pids 2>/dev/null
  for k in $(seq 1 30); do
    sleep 1
    pids=$(ours)
    [ -n "$pids" ] || return 0
  done
  kill -KILL $pids 2>/dev/null
}
trap stop_ours EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
port_busy() { (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

cd "$LINGBOT_ROOT" || exit 1
for i in $(seq 0 $((NGPU - 1))); do
  for p in $((START_PORT + i)) $((MASTER_PORT + i)); do
    if port_busy "$p"; then
      log "FATAL: port $p is in use; choose other START_PORT/MASTER_PORT or stop what holds it"
      exit 1
    fi
  done
done

log "curobo/warp preflight on GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" - <<'PY' || { log "FATAL: curobo warp kernels do not load"; exit 1; }
import warp as wp
wp.init()
import curobo.util.warp_interpolation as wi
wp.load_module(wi, device=wp.get_device("cuda:0"))
print("[preflight] curobo warp module OK", flush=True)
PY

# ---- servers -----------------------------------------------------------------------------------
batch=$(date +%Y%m%d_%H%M%S)
declare -a SRV_PID SRV_LOG
for i in $(seq 0 $((NGPU - 1))); do
  PORT=$((START_PORT + i)); MPORT=$((MASTER_PORT + i))
  SRV_LOG[$i]="$LOG_DIR/server_${ARM_TAG}_${i}_${batch}.log"
  QWAM_RUN_ID="$RUN_ID" CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" -m torch.distributed.run --nproc_per_node 1 --master_port "$MPORT" \
      wan_va/wan_va_server.py --config-name robotwin --save_root ./visualization/ \
      --port "$PORT" > "${SRV_LOG[$i]}" 2>&1 &
  SRV_PID[$i]=$!
  log "  server $i -> GPU ${GPUS[$i]} port $PORT (pid ${SRV_PID[$i]})"
  sleep 4
done
for i in $(seq 0 $((NGPU - 1))); do
  PORT=$((START_PORT + i)); waited=0
  until port_busy "$PORT"; do
    if ! kill -0 "${SRV_PID[$i]}" 2>/dev/null; then
      log "FATAL: server $i exited during startup; see ${SRV_LOG[$i]}"; exit 1
    fi
    if [ "$waited" -ge "$SERVER_TIMEOUT" ]; then
      log "FATAL: server $i not listening after ${SERVER_TIMEOUT}s; see ${SRV_LOG[$i]}"; exit 1
    fi
    sleep 10; waited=$((waited + 10))
  done
  log "  server $i ready"
done
# The model is loaded (and the checkpoint installed) before a server listens.
for i in $(seq 0 $((NGPU - 1))); do
  if [ -n "${LB_QUANT_CKPT:-}" ]; then
    grep -Eq '\[qwam\] replaced ([0-9]+)/\1 Linears' "${SRV_LOG[$i]}" || {
      log "FATAL: server $i did not install $LB_QUANT_CKPT; see ${SRV_LOG[$i]}"; exit 1; }
  elif grep -q '\[qwam\] installing' "${SRV_LOG[$i]}"; then
    log "FATAL: bf16 run but server $i installed a checkpoint; see ${SRV_LOG[$i]}"; exit 1
  fi
done
log "all $NGPU servers ready ($([ -n "${LB_QUANT_CKPT:-}" ] && echo "W4A4 checkpoint installed" || echo bf16))"

# ---- work queue --------------------------------------------------------------------------------
declare -a GPU_PID GPU_TASK BAD DEAD
for i in $(seq 0 $((NGPU - 1))); do GPU_PID[$i]=0; GPU_TASK[$i]=""; BAD[$i]=0; DEAD[$i]=0; done
next=0; done_n=0; fail_n=0; FAILED_TASKS=""

launch_on() {       # $1 = slot, $2 = task
  local g=$1 t=$2 port=$((START_PORT + $1))
  local lf="$LOG_DIR/${t}_${ARM_TAG}_$(date +%Y%m%d_%H%M%S).log"
  QWAM_RUN_ID="$RUN_ID" CUDA_VISIBLE_DEVICES="${GPUS[$g]}" PYTHONWARNINGS=ignore::UserWarning \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -m evaluation.robotwin.eval_polict_client_openpi \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name "$t" --task_config "$TASK_CONFIG" --train_config_name 0 --model_name 0 \
      --ckpt_setting "$ARM_TAG" --seed 0 --policy_name ACT --save_root "$RES" \
      --video_guidance_scale 5 --action_guidance_scale 1 --test_num "$TEST_NUM" \
      --port "$port" > "$lf" 2>&1 &
  GPU_PID[$g]=$!; GPU_TASK[$g]="$t"
  log "  launch $t -> slot $g (GPU ${GPUS[$g]}, pid ${GPU_PID[$g]}) [$((next + 1))/${#TASKS[@]}]"
}

# A task counts as done only if it wrote episodes; a slot whose tasks write nothing twice in a row
# (e.g. a broken CUDA context in its server) is taken out of the rotation.
while [ $((done_n + fail_n)) -lt "${#TASKS[@]}" ]; do
  for i in $(seq 0 $((NGPU - 1))); do
    if [ "${GPU_PID[$i]}" -ne 0 ] && ! kill -0 "${GPU_PID[$i]}" 2>/dev/null; then
      t="${GPU_TASK[$i]}"
      n=$(find "$RES"/stseed-*/visualization/"$t" -name '*.mp4' 2>/dev/null | wc -l)
      if [ "$n" -eq 0 ]; then
        fail_n=$((fail_n + 1)); BAD[$i]=$((BAD[$i] + 1)); FAILED_TASKS="$FAILED_TASKS $t"
        log "  FAILED $t (slot $i) wrote 0 episodes"
        if [ "${BAD[$i]}" -ge 2 ]; then
          DEAD[$i]=1; log "  slot $i disabled after 2 consecutive zero-episode tasks"
        fi
      else
        BAD[$i]=0; done_n=$((done_n + 1))
        log "  done $t (slot $i, $n episodes) [$done_n/${#TASKS[@]}]"
      fi
      GPU_PID[$i]=0; GPU_TASK[$i]=""
    fi
    if [ "${GPU_PID[$i]}" -eq 0 ] && [ "${DEAD[$i]}" -eq 0 ] && [ "$next" -lt "${#TASKS[@]}" ]; then
      launch_on "$i" "${TASKS[$next]}"; next=$((next + 1))
    fi
  done
  alive=0
  for i in $(seq 0 $((NGPU - 1))); do [ "${DEAD[$i]}" -eq 0 ] && alive=$((alive + 1)); done
  if [ "$alive" -eq 0 ]; then log "FATAL: every slot is disabled"; break; fi
  busy=0
  for i in $(seq 0 $((NGPU - 1))); do [ "${GPU_PID[$i]}" -ne 0 ] && busy=$((busy + 1)); done
  if [ "$busy" -eq 0 ] && [ "$next" -ge "${#TASKS[@]}" ]; then break; fi
  sleep 20
done

# Completion is judged from the episodes on disk.
verified=0
for t in "${TASKS[@]}"; do
  [ "$(episodes_done "$t")" -ge "$TEST_NUM" ] && verified=$((verified + 1))
done
if [ "$verified" -eq "${#TASKS[@]}" ]; then
  log "======== $ARM_TAG: all ${#TASKS[@]} tasks complete ========"
  exit 0
fi
log "======== $ARM_TAG: INCOMPLETE, $verified/${#TASKS[@]} tasks have >= $TEST_NUM episodes ========"
[ -n "$FAILED_TASKS" ] && log "  zero-episode tasks:$FAILED_TASKS"
log "  rerun the same command to resume"
exit 1
