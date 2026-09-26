"""Merge the shards of estimate_aog.py.

Action mode: the float64 sums are added over the shards (in file-name order) and divided by the
number of frames, and the dense per-layer AOGs are stored in float32 as {layer: [d_in, d_in]}, the
file Q-WAM loads (QWAM_AOG). A JSON sidecar records the calibration protocol.
Video-mass mode: stores {"tr": {layer: float}, "diag": {layer: [d_in]}, "n_frames", "nprobe"} for
action_mass.py.

  python scripts/fastwam/merge_aog.py --shard-dir work/fastwam/aog_shards --out work/fastwam/aog_action.pt
"""
import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the result even if some calibration episodes are missing")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.shard_dir, "aog_shard*.pt")))
    if not paths:
        raise SystemExit(f"no aog_shard*.pt in {args.shard_dir}")
    mode, protocol, n_frames, episodes = None, None, 0, []
    total, tr, diag = {}, {}, {}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=True)
        pr = {k: v for k, v in d["protocol"].items() if k not in ("episodes", "shard")}
        if mode is None:
            mode, protocol = d["mode"], pr
        elif d["mode"] != mode or pr != protocol:
            raise SystemExit(f"{p} does not match the other shards (mode or protocol differs)")
        n_frames += int(d["n_frames"])
        episodes += list(d["protocol"]["episodes"])
        if mode == "action":
            for n, v in d["aog_sum"].items():
                if n in total:
                    total[n] += v.double()
                else:
                    total[n] = v.double()
        else:
            for n, v in d["tr_sum"].items():
                tr[n] = tr.get(n, 0.0) + float(v)
            for n, v in d["diag_sum"].items():
                diag[n] = v.double() if n not in diag else diag[n] + v.double()
        del d
        print(f"  {os.path.basename(p)}")

    missing = sorted(set(protocol["all_episodes"]) - set(episodes))
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} calibration episodes have no shard (e.g. {missing[:5]}); "
                         f"run the missing shards or pass --allow-partial")
    if n_frames == 0:
        raise SystemExit("the shards hold no frames")
    protocol.update(n_frames=n_frames, episodes=sorted(episodes), n_shards_merged=len(paths))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if mode == "action":
        aog = {n: (v / n_frames).float() for n, v in total.items()}
        torch.save(aog, args.out)
        summary = f"{len(aog)} dense action-expert AOGs"
    else:
        torch.save({"tr": {n: v / n_frames for n, v in tr.items()},
                    "diag": {n: (v / n_frames).float() for n, v in diag.items()},
                    "n_frames": n_frames, "nprobe": protocol["nprobe"]}, args.out)
        summary = f"video-expert trace/diagonal for {len(tr)} layers"
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"mode": mode, **protocol}, f, indent=1)
    print(f"merged {len(paths)} shards: {len(set(episodes))} episodes, {n_frames} frames, {summary} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
