"""Dump the calibration set as raw per-camera frames for models that cannot decode AV1 video.

Run in the FastWAM environment (it needs torchcodec). Each calibration episode is written to
<out-dir>/ep<episode>.npz with uint8 frames per camera (head, left, right), the raw 14-d state,
the frame indices and the instruction, so ImageWAM and LingBot-VA can apply their own
preprocessing to exactly the same frames.

  FASTWAM_ROOT=/path/to/FastWAM python scripts/common/dump_c50_frames.py --out-dir <dir> [--seed 42]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam import robotwin_calib as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="1 = all frames (the calibration protocol)")
    ap.add_argument("--wh", type=str, default="320x240", help="per-camera resize, WxH")
    ap.add_argument("--seed", type=int, default=None, help=f"episode-sampling seed (default {C.CALIB_SEED})")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    import cv2
    import pandas as pd

    W, H = (int(x) for x in args.wh.lower().split("x"))
    seed = C.CALIB_SEED if args.seed is None else args.seed
    os.makedirs(args.out_dir, exist_ok=True)
    eps = C.select_episodes_random(per_task=1, num_tasks=50, seed=seed)
    instr = C._load_instructions()
    print(f"[dump] seed={seed} episodes={len(eps)} stride={args.stride} size={W}x{H}", flush=True)

    total = 0
    for k, ep in enumerate(eps):
        fp = os.path.join(args.out_dir, f"ep{ep:06d}.npz")
        if os.path.exists(fp):
            print(f"[dump] {k+1}/{len(eps)} ep={ep} exists, skip", flush=True)
            continue
        df = pd.read_parquet(C._parquet(ep), columns=["observation.state"])
        n = len(df)
        idxs = list(range(0, n, args.stride))
        cams = {}
        for key, short in zip(C.VIDEO_KEYS, ("head", "left", "right")):
            vid = C._video(ep, key)
            cams[short] = np.stack([
                cv2.resize(C._decode_frame(vid, i), (W, H)).astype(np.uint8) for i in idxs])
        state = np.stack([np.asarray(df["observation.state"].iloc[i], dtype=np.float32)[:14]
                          for i in idxs])
        np.savez_compressed(fp, head=cams["head"], left=cams["left"], right=cams["right"],
                            state=state, episode=ep, frames=np.asarray(idxs),
                            prompt=str(instr.get(ep, "Do the manipulation task.")))
        total += len(idxs)
        print(f"[dump] {k+1}/{len(eps)} ep={ep} task={ep // C.EPISODES_PER_TASK} "
              f"+{len(idxs)} frames, total {total}", flush=True)
    print(f"[dump] done: {total} frames across {len(eps)} episodes -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
