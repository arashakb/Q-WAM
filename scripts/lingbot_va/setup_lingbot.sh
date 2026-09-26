#!/usr/bin/env bash
# Prepare $LINGBOT_ROOT for Q-WAM: lingbot-va at the pinned commit with the Q-WAM patches, RoboTwin
# 2.0 at the pinned commit (nested in $LINGBOT_ROOT/RoboTwin) with the install edits of the
# lingbot-va README, and the RoboTwin post-trained weights at the pinned revision. Idempotent.
#
#   LINGBOT_ROOT=<dir> bash scripts/lingbot_va/setup_lingbot.sh
#
#   LINGBOT_CKPT     weight directory (default $LINGBOT_ROOT/checkpoints/lingbot-va-posttrain-robotwin)
#   SKIP_WEIGHTS=1   do not download the weights
#
# The environment itself (Python 3.10, torch 2.9.0+cu126, diffusers 0.36.0, transformers 4.55.2,
# RoboTwin requirements, pytorch3d, curobo) is installed as described in docs/lingbot_va.md.
set -euo pipefail
QWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCHES="$QWAM_ROOT/patches/lingbot_va"
: "${LINGBOT_ROOT:?set LINGBOT_ROOT to the directory for the lingbot-va checkout}"

LINGBOT_URL=https://github.com/Robbyant/lingbot-va.git
LINGBOT_COMMIT=58c2ae5bac46bd8114065bea9d7d256eb67c16c3
ROBOTWIN_URL=https://github.com/RoboTwin-Platform/RoboTwin.git
ROBOTWIN_COMMIT=2eeec322d95799f537cbfe5f291a8220d965ccb8
WEIGHTS_REPO=robbyant/lingbot-va-posttrain-robotwin
WEIGHTS_REV=8c9dea8abbc5c91cc9e18bc3264b8915083bbe70
LINGBOT_CKPT="${LINGBOT_CKPT:-$LINGBOT_ROOT/checkpoints/lingbot-va-posttrain-robotwin}"

checkout() {        # $1 = url, $2 = dir, $3 = commit
  if [ ! -e "$2/.git" ]; then
    if [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
      echo "ERROR: $2 is neither empty nor a git checkout" >&2; exit 1
    fi
    git clone "$1" "$2"
    git -C "$2" checkout -q "$3"
  fi
  local head
  head=$(git -C "$2" rev-parse HEAD)
  [ "$head" = "$3" ] || echo "WARNING: $2 is at $head, the paper runs used $3" >&2
}

apply_patch() {     # $1 = repo, $2 = patch; applied once, re-runs are no-ops
  if git -C "$1" apply --check "$2" 2>/dev/null; then
    git -C "$1" apply "$2"; echo "applied  $(basename "$2")"
  elif git -C "$1" apply -R --check "$2" 2>/dev/null; then
    echo "present  $(basename "$2")"
  else
    echo "ERROR: $(basename "$2") does not apply to $1" >&2; exit 1
  fi
}

checkout "$LINGBOT_URL" "$LINGBOT_ROOT" "$LINGBOT_COMMIT"
for p in wan_va_server va_robotwin_cfg eval_polict_client_openpi; do
  apply_patch "$LINGBOT_ROOT" "$PATCHES/$p.patch"
done

checkout "$ROBOTWIN_URL" "$LINGBOT_ROOT/RoboTwin" "$ROBOTWIN_COMMIT"
apply_patch "$LINGBOT_ROOT/RoboTwin" "$PATCHES/robotwin_install.patch"

if [ -z "${SKIP_WEIGHTS:-}" ] && [ ! -f "$LINGBOT_CKPT/transformer/config.json" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf download "$WEIGHTS_REPO" --revision "$WEIGHTS_REV" --local-dir "$LINGBOT_CKPT"
  else
    huggingface-cli download "$WEIGHTS_REPO" --revision "$WEIGHTS_REV" --local-dir "$LINGBOT_CKPT"
  fi
fi
if [ -f "$LINGBOT_CKPT/transformer/config.json" ] \
   && ! grep -Eq '"attn_mode": *"(torch|flashattn)"' "$LINGBOT_CKPT/transformer/config.json"; then
  echo "WARNING: set \"attn_mode\" to \"torch\" in $LINGBOT_CKPT/transformer/config.json for inference" >&2
fi

cat <<EOF

lingbot-va ready at $LINGBOT_ROOT (weights: $LINGBOT_CKPT).
Remaining one-time steps, inside the LingBot-VA environment:
  cd $LINGBOT_ROOT/RoboTwin && bash script/_install.sh && bash script/_download_assets.sh
Then export LINGBOT_ROOT=$LINGBOT_ROOT$( [ "$LINGBOT_CKPT" = "$LINGBOT_ROOT/checkpoints/lingbot-va-posttrain-robotwin" ] || echo " LINGBOT_CKPT=$LINGBOT_CKPT" )
EOF
