"""Action mass of each Fast-WAM expert (paper Eq. mass), the quantity that selects the expert for ASP.

    mu_E = sum_{l in E} tr(G~_l),   G~_l = H diag(s) G_l diag(s) H

H is orthogonal, so tr(G~_l) = sum_i s_i^2 (G_l)_ii and only the diagonal of each AOG is needed:
the dense action-expert AOGs (merge_aog.py) and the video-expert diagonals (estimate_aog.py
--mode video-mass, then merge_aog.py). s is the SmoothQuant factor (alpha 0.5) computed from the
activation absmax and the checkpoint weights. The stored AOGs are sums over the probes of each
frame, so each expert's mass is divided by its number of probes per frame before comparing.

  python scripts/fastwam/action_mass.py --aog work/fastwam/aog_action.pt \
      --video-mass work/fastwam/aog_video_mass.pt --absmax work/fastwam/absmax.pt \
      --ckpt $FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam.fastwam.checkpoint import block_linears  # noqa: E402
from qwam.fastwam.install import ACTION_EXPERT, load_absmax, load_aog  # noqa: E402
from qwam.quant import compute_smooth  # noqa: E402


def smoothed_trace(diag: torch.Tensor, absmax: torch.Tensor, weight: torch.Tensor, alpha: float) -> float:
    """tr(H diag(s) G diag(s) H) = sum_i s_i^2 G_ii."""
    s = compute_smooth(absmax, weight.float(), alpha)
    return float((s.pow(2) * diag.float()).sum())


def probes_of(path: str, stored, default: int) -> int:
    """Probes per frame of an AOG file: from the file, its JSON sidecar, or the default."""
    if isinstance(stored, dict) and "nprobe" in stored:
        return int(stored["nprobe"])
    sidecar = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            return int(json.load(f)["nprobe"])
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aog", required=True, help="dense action-expert AOGs (merge_aog.py)")
    ap.add_argument("--video-mass", required=True, help="video-expert trace/diagonal (merge_aog.py)")
    ap.add_argument("--absmax", required=True, help="activation absmax (merge_absmax.py)")
    ap.add_argument("--ckpt", required=True, help="Fast-WAM checkpoint (for the weights in s)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--aog-probes", type=int, default=None, help="default: from the files, else 12")
    ap.add_argument("--video-probes", type=int, default=None, help="default: from the files, else 8")
    args = ap.parse_args()

    weights = block_linears(args.ckpt)
    absmax = load_absmax(args.absmax)
    aog = load_aog(args.aog)
    video = torch.load(args.video_mass, map_location="cpu", mmap=True, weights_only=True)
    probes = {"action": args.aog_probes or probes_of(args.aog, None, 12),
              "video": args.video_probes or probes_of(args.video_mass, video, 8)}

    diags = {n: g.diagonal() for n, g in aog.items()}
    diags.update(video["diag"])
    mass, zero = {}, {}
    for name, d in diags.items():
        e = "action" if name.startswith(ACTION_EXPERT) else "video"
        m = smoothed_trace(d, absmax[name], weights[name][0], args.alpha) / probes[e]
        mass[e] = mass.get(e, 0.0) + m
        zero[e] = zero.get(e, 0) + (m == 0.0)
    total = sum(mass.values())
    for e in sorted(mass, key=mass.get, reverse=True):
        n_layers = sum(1 for n in diags if n.startswith(ACTION_EXPERT) == (e == "action"))
        print(f"{e:7s} expert: mu = {mass[e]:.6g} ({100 * mass[e] / total:.2f}%), {n_layers} layers, "
              f"{zero[e]} with zero mass, {probes[e]} probes/frame")
    print(f"ASP protects the {max(mass, key=mass.get)} expert "
          f"({max(mass.values()) / max(min(mass.values()), 1e-30):.1f}x the mass of the other)")


if __name__ == "__main__":
    main()
