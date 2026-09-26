"""Merge the absmax shards of calibrate_absmax.py into the cache Q-WAM loads (elementwise max).

  python scripts/fastwam/merge_absmax.py --shard-dir work/fastwam/absmax_shards --out work/fastwam/absmax.pt
"""
import argparse
import glob
import os

import torch

EXPECTED_LAYERS = 600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the cache even if some calibration episodes are missing")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.shard_dir, "absmax_shard*.pt")))
    if not paths:
        raise SystemExit(f"no absmax_shard*.pt in {args.shard_dir}")
    absmax, n_frames, episodes, protocol = {}, 0, [], None
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        pr = {k: v for k, v in d["protocol"].items() if k not in ("episodes", "shard")}
        if protocol is None:
            protocol = pr
        elif pr != protocol:
            raise SystemExit(f"{p} was calibrated with a different protocol:\n  {pr}\nvs\n  {protocol}")
        n_frames += int(d["n_frames"])
        episodes += list(d["protocol"]["episodes"])
        for n, v in d["absmax"].items():
            absmax[n] = v if n not in absmax else torch.maximum(absmax[n], v)
        print(f"  {os.path.basename(p)}: {d['n_frames']} frames")

    missing = sorted(set(protocol["all_episodes"]) - set(episodes))
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} calibration episodes have no shard (e.g. {missing[:5]}); "
                         f"run the missing shards or pass --allow-partial")
    if len(absmax) != EXPECTED_LAYERS:
        print(f"WARNING: {len(absmax)} layers, expected {EXPECTED_LAYERS} for Fast-WAM")
    protocol.update(n_frames=n_frames, episodes=sorted(episodes))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"absmax": absmax, "n_frames": n_frames, "protocol": protocol}, args.out)
    print(f"merged {len(paths)} shards: {len(set(episodes))} episodes, {n_frames} frames, "
          f"{len(absmax)} layers -> {args.out}")


if __name__ == "__main__":
    main()
