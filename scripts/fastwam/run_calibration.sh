#!/usr/bin/env bash
# Q-WAM calibration of Fast-WAM, sharded over the GPUs of one machine, then merged.
#
#   bash scripts/fastwam/run_calibration.sh [absmax|aog|video-mass|all]      (default: all)
#
#   absmax      activation absmax, 8 shards                 -> $QWAM_WORK/absmax.pt
#   aog         action-expert AOGs, 11 shards               -> $QWAM_WORK/aog_action.pt
#   video-mass  video-expert AOG trace/diagonal, 8 shards   -> $QWAM_WORK/aog_video_mass.pt
#               (only for scripts/fastwam/action_mass.py; not needed to run Q-WAM)
#   all         absmax, then aog
#
# Environment: FASTWAM_ROOT (required), GPUS (default: all GPUs; one shard per GPU at a time),
# QWAM_WORK (default: <this repository>/work/fastwam), PYTHON (default: python).
# The shard counts are those of the released calibration: the AOG probe seeds depend on the shard
# index, so --num-shards 11 reproduces the released AOG. An AOG shard needs ~43 GB of GPU memory.
# Finished shards are kept and skipped when the script is run again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWAM_ROOT="$(cd "$HERE/../.." && pwd)"
: "${FASTWAM_ROOT:?set FASTWAM_ROOT to the FastWAM checkout}"
export FASTWAM_ROOT
STAGE="${1:-all}"
PYTHON="${PYTHON:-python}"
QWAM_WORK="${QWAM_WORK:-$QWAM_ROOT/work/fastwam}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)}"
read -r -a GPU_IDS <<< "${GPUS:-0}"

run_shards() {  # script num_shards out_dir shard_prefix [script args...]
  local script="$1" n="$2" dir="$3" prefix="$4"; shift 4
  mkdir -p "$dir/logs"
  local queue="$dir/.queue"
  rm -rf "$queue"; mkdir -p "$queue"; echo 0 > "$queue/idx"
  echo ">>> $script: $n shards on GPUs [${GPU_IDS[*]}] -> $dir"
  for gpu in "${GPU_IDS[@]}"; do
    (
      while true; do
        exec 9>"$queue/lock"; flock 9
        i="$(cat "$queue/idx")"; echo $((i + 1)) > "$queue/idx"
        flock -u 9; exec 9>&-
        [ "$i" -ge "$n" ] && break
        if [ -f "$dir/${prefix}_shard$i.pt" ]; then echo "  shard $i exists, skipped"; continue; fi
        echo "  shard $i/$n on GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HERE/$script" --shard "$i" --num-shards "$n" \
          --out-dir "$dir" --device cuda:0 "$@" > "$dir/logs/shard$i.log" 2>&1 \
          || echo "  shard $i FAILED (see $dir/logs/shard$i.log)"
      done
    ) &
  done
  wait
  rm -rf "$queue"
}

stage_absmax() {
  run_shards calibrate_absmax.py 8 "$QWAM_WORK/absmax_shards" absmax
  "$PYTHON" "$HERE/merge_absmax.py" --shard-dir "$QWAM_WORK/absmax_shards" --out "$QWAM_WORK/absmax.pt"
}
stage_aog() {
  run_shards estimate_aog.py 11 "$QWAM_WORK/aog_shards" aog --mode action
  "$PYTHON" "$HERE/merge_aog.py" --shard-dir "$QWAM_WORK/aog_shards" --out "$QWAM_WORK/aog_action.pt"
}
stage_video_mass() {
  run_shards estimate_aog.py 8 "$QWAM_WORK/video_mass_shards" aog --mode video-mass
  "$PYTHON" "$HERE/merge_aog.py" --shard-dir "$QWAM_WORK/video_mass_shards" --out "$QWAM_WORK/aog_video_mass.pt"
}

case "$STAGE" in
  absmax)     stage_absmax ;;
  aog)        stage_aog ;;
  video-mass) stage_video_mass ;;
  all)        stage_absmax; stage_aog ;;
  *) echo "unknown stage '$STAGE' (absmax|aog|video-mass|all)" >&2; exit 1 ;;
esac
