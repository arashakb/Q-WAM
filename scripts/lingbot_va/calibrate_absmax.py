"""Activation absmax of the LingBot-VA block Linears over the calibration frames, one shard.

  LINGBOT_ROOT=<lingbot-va> python scripts/lingbot_va/calibrate_absmax.py \
      --raw-dir <dump_c50_frames.py output> --nshard 7 --shard <i>

Paper setting: 7 shards (one GPU each, VAE and text encoder resident on the GPU), merged with
merge_absmax.py. With --nshard > 1 the output is <out stem>_shard<i>.pt. An existing output is kept
unless --overwrite is given; interrupted runs leave only <out stem>.partial.pt behind.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "work" / "lingbot_va" / "lingbot_va_act_absmax.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.environ.get("C50_RAW"),
                    help="directory of ep<episode>.npz calibration dumps [C50_RAW]")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="absmax file (before the shard suffix)")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--offload", action="store_true",
                    help="upstream offload mode: VAE and text encoder run on the CPU")
    ap.add_argument("--no-park-text-encoder", dest="park", action="store_false", default=None,
                    help="keep the text encoder on the GPU between episodes (default: parked on "
                         "the CPU and moved in once per episode, unless --offload)")
    ap.add_argument("--empty-every", type=int, default=1, help="torch.cuda.empty_cache() every N frames")
    ap.add_argument("--ckpt-every", type=int, default=250, help="partial snapshot every N frames")
    ap.add_argument("--max-run-skips", type=int, default=25, help="abort after N consecutive OOM frames")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after N frames (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.raw_dir:
        ap.error("--raw-dir (or C50_RAW) is required")
    if not 0 <= args.shard < args.nshard:
        ap.error(f"--shard must be in [0, {args.nshard})")

    out = Path(args.out)
    if args.nshard > 1:
        out = out.with_name(f"{out.stem}_shard{args.shard}.pt")
    if out.exists() and not args.overwrite:
        print(f"[calib] {out} exists; keeping it (pass --overwrite to recompute)", flush=True)
        return 0
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from qwam.lingbot_va.calib import calibrate_absmax
    from qwam.lingbot_va.harness import build_server, calib_files

    files = calib_files(args.raw_dir)[args.shard::args.nshard]
    server, _ = build_server("robotwin_i2av", enable_offload=args.offload)
    park = (not args.offload) if args.park is None else args.park
    try:
        calibrate_absmax(server, files, out, empty_every=args.empty_every,
                         ckpt_every=args.ckpt_every, max_run_skips=args.max_run_skips,
                         park_text_encoder=park, max_frames=args.max_frames)
    except RuntimeError as e:
        print(f"[calib] FAILED: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
