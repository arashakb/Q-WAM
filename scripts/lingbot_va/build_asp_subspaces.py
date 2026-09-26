"""Rank-r ASP subspaces of LingBot-VA from the sketched AOG (method: qwam/lingbot_va/aog.py).

  LINGBOT_ROOT=<lingbot-va> python scripts/lingbot_va/build_asp_subspaces.py \
      --raw-dir <dump_c50_frames.py output> \
      [--absmax artifacts/lingbot_va/lingbot_va_act_absmax.pt] \
      [--out work/lingbot_va/lingbot_va_asp_subspaces_r32.pt]

Paper setting (the defaults): rank 32 with 32 extra sketch columns, 4 frames per episode (200
frames), one Gaussian probe per action call, seed 42, alpha 0.5, Hadamard block <= 1024, text encoder
parked on the CPU between episodes. One 80 GB GPU: about 1 h, 59 GB peak.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_ABSMAX = ROOT / "artifacts" / "lingbot_va" / "lingbot_va_act_absmax.pt"
DEFAULT_OUT = ROOT / "work" / "lingbot_va" / "lingbot_va_asp_subspaces_r32.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.environ.get("C50_RAW"),
                    help="directory of ep<episode>.npz calibration dumps [C50_RAW]")
    ap.add_argument("--absmax", default=str(DEFAULT_ABSMAX),
                    help="activation absmax defining the smoothing factors of the basis")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--oversample", type=int, default=32, help="extra sketch columns")
    ap.add_argument("--per-ep", type=int, default=int(os.environ.get("LB_PER_EP", "4")),
                    help="frames per calibration episode [LB_PER_EP]")
    ap.add_argument("--nprobe", type=int, default=1, help="Gaussian probes per action call")
    ap.add_argument("--seed", type=int, default=42, help="seed of the per-layer test matrices")
    ap.add_argument("--offload", action="store_true",
                    help="upstream offload mode: VAE and text encoder run on the CPU")
    ap.add_argument("--no-park-text-encoder", dest="park", action="store_false", default=True,
                    help="keep the text encoder on the GPU between episodes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.raw_dir:
        ap.error("--raw-dir (or C50_RAW) is required")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"[aog] {out} exists; keeping it (pass --overwrite to recompute)", flush=True)
        return 0
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from qwam.lingbot_va.aog import accumulate_sketches, build_subspace_file
    from qwam.lingbot_va.export import target_linears
    from qwam.lingbot_va.harness import build_server, calib_files

    absmax = torch.load(args.absmax, map_location="cpu", weights_only=True)
    files = calib_files(args.raw_dir)
    server, _ = build_server("robotwin_i2av", enable_offload=args.offload)
    sketch, trG, stats = accumulate_sketches(
        server, files, rank=args.rank, oversample=args.oversample, per_ep=args.per_ep,
        nprobe=args.nprobe, seed=args.seed, park_text_encoder=args.park)

    # The subspace construction is CPU linear algebra; release the GPU copy first.
    tr = server.transformer
    tr.to("cpu")
    torch.cuda.empty_cache()
    blob = build_subspace_file(sketch, trG, stats, target_linears(tr), absmax, rank=args.rank,
                               oversample=args.oversample, per_ep=args.per_ep,
                               nprobe=args.nprobe, seed=args.seed)
    blob["meta"]["absmax_file"] = Path(args.absmax).name
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(blob["meta"], indent=1))
    print(f"[aog] wrote {out}: {len(blob['subspaces'])} layers with V, "
          f"{blob['meta']['skipped_layers']} skipped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
