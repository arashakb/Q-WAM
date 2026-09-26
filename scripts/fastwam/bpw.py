"""Bits per weight (BPW) of Q-WAM on Fast-WAM, from the layer shapes of the checkpoint.

Over the 600 targeted block linears (5,913,968,640 weights), each layer [d_out, d_in] stores
    4 bits per weight                       INT4 weight
    16 bits per group of 32 weights         one scale per group along d_in
    16 bits x d_in                          smoothing vector (the rotation stores nothing)
    16 bits x d_out                         bias
and each ASP layer (action expert) also stores
    16 bits x rank x (d_in + d_out)         the basis V [d_in, r] and W~V [d_out, r]
BPW is the total divided by the number of weights: 4.6220 for the paper configuration.

  python scripts/fastwam/bpw.py --ckpt $FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
"""
import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam.fastwam.checkpoint import block_linears  # noqa: E402
from qwam.fastwam.install import ACTION_EXPERT, QWAMConfig  # noqa: E402


def main():
    cfg = QWAMConfig()
    default_ckpt = (os.path.join(os.environ["FASTWAM_ROOT"], "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt")
                    if os.environ.get("FASTWAM_ROOT") else None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=default_ckpt, required=default_ckpt is None)
    ap.add_argument("--rank", type=int, default=cfg.rank, help="ASP rank on the action expert (0 = none)")
    ap.add_argument("--w-bits", type=int, default=cfg.w_bits)
    ap.add_argument("--w-group", type=int, default=cfg.w_group)
    args = ap.parse_args()

    layers = block_linears(args.ckpt)
    parts = {"weights": 0, "group scales": 0, "smoothing vectors": 0, "biases": 0, "ASP branch": 0}
    n_weights = n_asp = 0
    for name, (w, b) in layers.items():
        d_out, d_in = w.shape
        n_weights += d_out * d_in
        parts["weights"] += args.w_bits * d_out * d_in
        parts["group scales"] += 16 * d_out * math.ceil(d_in / args.w_group)
        parts["smoothing vectors"] += 16 * d_in
        parts["biases"] += 16 * d_out if b is not None else 0
        if args.rank > 0 and name.startswith(ACTION_EXPERT):
            parts["ASP branch"] += 16 * args.rank * (d_in + d_out)
            n_asp += 1
    print(f"{len(layers)} block linears ({n_asp} with ASP rank {args.rank}), {n_weights:,} weights")
    for k, v in parts.items():
        print(f"  {k:18s} {v / n_weights:7.4f} bits per weight")
    total = sum(parts.values())
    print(f"BPW = {total / n_weights:.4f}")


if __name__ == "__main__":
    main()
