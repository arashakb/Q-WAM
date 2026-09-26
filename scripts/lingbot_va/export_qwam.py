"""Export a packed W4A4 LingBot-VA checkpoint. Defaults are the paper configuration (Q-WAM, Table 1).

  LINGBOT_ROOT=<lingbot-va> python scripts/lingbot_va/export_qwam.py --out <ckpt.pt>

W4A4, weight and activation group 32, alpha 0.5, Hadamard block <= 1024, ASP rank 32. Component
ablation (Table 3), all at group 32:
  per-group W4A4              --subspaces none --no-smooth --no-rotate
  + smoothing and rotation    --subspaces none
  + ASP (Q-WAM)               defaults

Runs on the CPU (about 15 GB of RAM, a minute or two); only the transformer weights are loaded.
Options in brackets can also be given through the environment.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_ABSMAX = ROOT / "artifacts" / "lingbot_va" / "lingbot_va_act_absmax.pt"
DEFAULT_SUBSPACES = ROOT / "artifacts" / "lingbot_va" / "lingbot_va_asp_subspaces_r32.pt"


def _env_flag(name, default=True):
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("LB_OUT"), help="output checkpoint [LB_OUT]")
    ap.add_argument("--absmax", default=os.environ.get("LB_ABSMAX", str(DEFAULT_ABSMAX)),
                    help="activation absmax (merge_absmax.py) [LB_ABSMAX]")
    ap.add_argument("--subspaces", default=os.environ.get("LB_ASP_SUBSPACES", str(DEFAULT_SUBSPACES)),
                    help="ASP subspace file (build_asp_subspaces.py), or 'none' for rank 0 "
                         "[LB_ASP_SUBSPACES]")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("LB_ASP_RANK", "32")),
                    help="ASP rank; 0 disables ASP [LB_ASP_RANK]")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false", default=_env_flag("LB_SMOOTH"),
                    help="disable smoothing, s = 1 [LB_SMOOTH=0]")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false", default=_env_flag("LB_ROTATE"),
                    help="disable the Hadamard rotation [LB_ROTATE=0]")
    ap.add_argument("--lingbot-ckpt", default=None,
                    help="base model directory (default: $LINGBOT_CKPT via the patched upstream config)")
    args = ap.parse_args()
    if not args.out:
        ap.error("--out (or LB_OUT) is required")

    import torch

    from qwam.lingbot_va.export import ExportConfig, export_checkpoint, tensor_digest
    from qwam.lingbot_va.harness import load_transformer

    use_asp = args.subspaces not in (None, "", "none") and args.rank > 0
    cfg = ExportConfig(smooth=args.smooth, rotate=args.rotate, asp_rank=args.rank if use_asp else 0)
    absmax = None
    if cfg.smooth:
        absmax = torch.load(args.absmax, map_location="cpu", weights_only=True)
        print(f"[export] absmax {args.absmax}: {len(absmax)} layers", flush=True)
    subspaces, prov = None, {"absmax_file": Path(args.absmax).name if cfg.smooth else None,
                             "asp_subspaces": None, "asp_subspaces_digest": None}
    if use_asp:
        if not Path(args.subspaces).is_file():
            sys.exit(f"no subspace file at {args.subspaces}; build it with build_asp_subspaces.py "
                     f"or pass --subspaces none for rank 0")
        subspaces = torch.load(args.subspaces, map_location="cpu", weights_only=True)
        prov["asp_subspaces"] = Path(args.subspaces).name
        prov["asp_subspaces_digest"] = tensor_digest(
            {n: e["V"] for n, e in subspaces["subspaces"].items()})
        print(f"[export] subspaces {args.subspaces}", flush=True)

    model = load_transformer(args.lingbot_ckpt, device="cpu")
    try:
        ckpt = export_checkpoint(model, cfg, act_absmax=absmax, subspaces=subspaces, provenance=prov)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"[export] refused: {e}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(ckpt["meta"], indent=2))
    print(f"[export] wrote {out} ({out.stat().st_size / 2**30:.2f} GiB, {len(ckpt['layers'])} layers, "
          f"{ckpt['meta']['bpw_with_asp_branch']:.4f} BPW)", flush=True)


if __name__ == "__main__":
    main()
