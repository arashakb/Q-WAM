"""Merge per-shard activation absmax files by elementwise maximum.

  python scripts/lingbot_va/merge_absmax.py --expect 7 [--out work/lingbot_va/lingbot_va_act_absmax.pt]

Reads <out stem>_shard*.pt next to --out, or the shard files given as arguments. Each shard saw a
disjoint set of episodes, so the merge equals the statistic of one unsharded pass.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "work" / "lingbot_va" / "lingbot_va_act_absmax.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*", help="shard files (default: <out stem>_shard*.pt)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--expect", type=int, default=7, help="number of shards required")
    args = ap.parse_args()

    import torch

    from qwam.lingbot_va.calib import merge_shards

    out = Path(args.out)
    files = args.shards or glob.glob(str(out.with_name(f"{out.stem}_shard*.pt")))
    files = sorted(files, key=lambda f: int(m.group(1)) if (m := re.search(r"shard(\d+)", f)) else -1)
    if len(files) != args.expect:
        sys.exit(f"expected {args.expect} shard files, found {len(files)}: {[Path(f).name for f in files]}")

    shards, frames = [], 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        present = sum(1 for v in d.values() if torch.is_tensor(v))
        side = Path(f).with_suffix(".json")
        n = json.loads(side.read_text())["frames"] if side.exists() else None
        frames += n or 0
        print(f"  {Path(f).name:44s} {len(d):4d} layers, {present} with data"
              + (f", {n} frames" if n is not None else ""))
        shards.append(d)
    try:
        merged = merge_shards(shards)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)
    print(f"merged {len(files)} shards -> {len(merged)} layers, "
          f"{sum(v.numel() for v in merged.values())} channels"
          + (f", {frames} frames" if frames else "") + f" -> {out}")


if __name__ == "__main__":
    main()
