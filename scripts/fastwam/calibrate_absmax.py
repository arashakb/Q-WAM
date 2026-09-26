"""Per-input-channel activation absmax of Fast-WAM's 600 block linears (one shard).

The calibration set is one random demonstration episode per RoboTwin task (50 tasks, seed 42) and
every frame of each episode, 10,919 frames in total (qwam.robotwin_calib). Every frame runs the
bf16 model through Fast-WAM's inference (video prefill, then 10 action denoising steps with horizon
32), and a forward pre-hook keeps max |x| per input channel for every block linear of both experts.
Shards take the episode list round-robin; merge them with merge_absmax.py. The maximum does not
depend on the sharding.

  FASTWAM_ROOT=/path/to/FastWAM CUDA_VISIBLE_DEVICES=0 \
  python scripts/fastwam/calibrate_absmax.py --shard 0 --num-shards 8 --out-dir work/fastwam/absmax_shards
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam import robotwin_calib as C  # noqa: E402
from qwam.fastwam import target_linears  # noqa: E402
from qwam.fastwam.harness import ACTION_HORIZON, NUM_INFERENCE_STEPS  # noqa: E402


def register_absmax_hooks(targets: dict, store: dict) -> list:
    """Forward pre-hooks that keep the running per-input-channel max |x| of each layer in `store`."""
    def make(name):
        def hook(module, inputs):
            x = inputs[0].detach()
            cmax = x.reshape(-1, x.shape[-1]).float().abs().amax(0)
            store[name] = cmax if store[name] is None else torch.maximum(store[name], cmax)
        return hook
    return [m.register_forward_pre_hook(make(n)) for n, m in targets.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: the released RoboTwin checkpoint)")
    ap.add_argument("--seed", type=int, default=C.CALIB_SEED, help="episode-sampling seed")
    ap.add_argument("--per-task", type=int, default=C.CALIB_PER_TASK, help="episodes per task")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (1 = every frame)")
    ap.add_argument("--num-steps", type=int, default=NUM_INFERENCE_STEPS, help="denoising steps")
    args = ap.parse_args()

    from qwam.fastwam.harness import load_model
    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    eps = C.select_episodes_random(per_task=args.per_task, num_tasks=C.NUM_TASKS, seed=args.seed)
    mine = eps[args.shard::args.num_shards]
    tag = f"[absmax {args.shard}/{args.num_shards}]"
    print(f"{tag} {len(eps)} episodes (seed {args.seed}), {len(mine)} on this shard: {mine}", flush=True)

    model = load_model(device=dev, ckpt=args.ckpt)
    targets = target_linears(model)
    absmax = {n: None for n in targets}
    hooks = register_absmax_hooks(targets, absmax)
    mean, std, _ = C._load_state_stats()

    t0, n_frames = time.time(), 0
    for k, ep in enumerate(mine):
        frames = C.build_episode_allframes(ep, stride=args.stride, mean=mean, std=std)
        for o in frames:
            with torch.no_grad():
                model.infer_action(prompt=o["prompt"], input_image=o["image"].to(dev),
                                   action_horizon=ACTION_HORIZON, proprio=o["proprio"].to(dev),
                                   num_inference_steps=args.num_steps, seed=0, rand_device="cpu")
            n_frames += 1
        print(f"{tag} {k + 1}/{len(mine)} episode {ep} (task {ep // C.EPISODES_PER_TASK}) "
              f"+{len(frames)} frames, total {n_frames} ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()

    out = {
        "absmax": {n: v.cpu() for n, v in absmax.items() if v is not None},
        "n_frames": n_frames,
        "protocol": {"seed": args.seed, "per_task": args.per_task, "num_tasks": C.NUM_TASKS,
                     "stride": args.stride, "num_steps": args.num_steps, "shard": args.shard,
                     "num_shards": args.num_shards, "episodes": mine, "all_episodes": eps},
    }
    path = os.path.join(args.out_dir, f"absmax_shard{args.shard}.pt")
    torch.save(out, path)
    print(f"{tag} done: {n_frames} frames, {len(out['absmax'])} layers -> {path}", flush=True)


if __name__ == "__main__":
    main()
