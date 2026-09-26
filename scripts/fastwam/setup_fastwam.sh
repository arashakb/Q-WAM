#!/usr/bin/env bash
# Prepare a FastWAM checkout for Q-WAM: apply the policy hook and link the policy into RoboTwin.
#
#   FASTWAM_ROOT=/path/to/FastWAM bash scripts/fastwam/setup_fastwam.sh
#
# 1. checks that FastWAM is at the commit the patch was made against (warns otherwise);
# 2. applies patches/fastwam/deploy_policy.patch (QWAM_ENABLE hook in the RoboTwin policy), once;
# 3. links experiments/robotwin/fastwam_policy into third_party/RoboTwin/policy, as FastWAM's README does.
set -euo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${FASTWAM_ROOT:?set FASTWAM_ROOT to your FastWAM checkout}"
FASTWAM_ROOT="$(cd "$FASTWAM_ROOT" && pwd)"
PIN=45d8e1458921d83f8ad6cf9ce993d371208dabd0
PATCH="$QWAM_ROOT/patches/fastwam/deploy_policy.patch"
POLICY="$FASTWAM_ROOT/experiments/robotwin/fastwam_policy"
LINK="$FASTWAM_ROOT/third_party/RoboTwin/policy/fastwam_policy"

[ -f "$POLICY/deploy_policy.py" ] || { echo "not a FastWAM checkout: $FASTWAM_ROOT" >&2; exit 1; }

head="$(git -C "$FASTWAM_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "$head" != "$PIN" ]; then
  echo "WARNING: FastWAM is at $head; the patch and the reported results use $PIN." >&2
fi

cd "$FASTWAM_ROOT"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[setup] policy hook already applied"
elif git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
  echo "[setup] applied $(basename "$PATCH")"
else
  echo "cannot apply $PATCH to $POLICY/deploy_policy.py (modified file?)" >&2
  exit 1
fi

if [ -L "$LINK" ]; then
  [ "$(readlink -f "$LINK")" = "$(readlink -f "$POLICY")" ] || {
    echo "$LINK points to $(readlink -f "$LINK"), expected $POLICY" >&2; exit 1; }
elif [ -e "$LINK" ]; then
  echo "$LINK exists and is not a symlink; move it away first" >&2
  exit 1
else
  ln -s "$POLICY" "$LINK"
fi
echo "[setup] RoboTwin policy link: $LINK -> $POLICY"
echo "[setup] done. qwam is put on PYTHONPATH by scripts/fastwam/run_robotwin_eval.sh."
