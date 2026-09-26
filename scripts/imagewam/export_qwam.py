"""Export a W4A4 ImageWAM checkpoint. Defaults are the paper configuration (Q-WAM, Table 1).

  python scripts/imagewam/export_qwam.py --base-ckpt <model.pt> --absmax <absmax.pt> --out <ckpt.pt>

Component ablation (Table 3), all at weight/activation group 32:
  per-group W4A4              --no-smooth --no-rotate --subspaces none
  + smoothing and rotation    --subspaces none
  + ASP (Q-WAM)               defaults (shipped rank-32 subspace file)

Every option can also be given through the environment variable in brackets.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwam.imagewam.export import export_checkpoint  # noqa: E402

DEFAULT_SUBSPACES = ROOT / "artifacts" / "imagewam" / "imagewam_asp_subspaces_r32.pt"


def _env_flag(name, default=True):
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "false", "False", "")


def _default_base_ckpt():
    if os.environ.get("IW_BASE_CKPT"):
        return os.environ["IW_BASE_CKPT"]
    if os.environ.get("IMAGEWAM_ROOT"):
        return str(Path(os.environ["IMAGEWAM_ROOT"])
                   / "checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default=_default_base_ckpt(),
                    help="released ImageWAM RoboTwin model.pt [IW_BASE_CKPT; "
                         "default $IMAGEWAM_ROOT/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt]")
    ap.add_argument("--absmax", default=os.environ.get("IW_ABSMAX"),
                    help="merged activation absmax from calibrate_absmax.sh [IW_ABSMAX]")
    ap.add_argument("--out", default=os.environ.get("IW_OUT"), help="output checkpoint [IW_OUT]")
    ap.add_argument("--subspaces", default=os.environ.get("IW_ASP_SUBSPACES", str(DEFAULT_SUBSPACES)),
                    help="ASP subspace file, or 'none' for rank 0 [IW_ASP_SUBSPACES]")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("IW_ASP_RANK", "32")),
                    help="ASP rank; 0 disables ASP [IW_ASP_RANK]")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false",
                    default=_env_flag("IW_SMOOTH"), help="disable smoothing (s = 1) [IW_SMOOTH=0]")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    default=_env_flag("IW_ROTATE"), help="disable the Hadamard rotation [IW_ROTATE=0]")
    ap.add_argument("--w-group", type=int, default=int(os.environ.get("IW_WGROUP", "32")),
                    help="weight group size [IW_WGROUP]")
    ap.add_argument("--a-group", type=int, default=int(os.environ.get("IW_AGROUP", "0")),
                    help="activation group size, 0 = weight group [IW_AGROUP]")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5, help="smoothing strength")
    ap.add_argument("--fwht-block-max", type=int, default=1024, help="Hadamard block cap")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    a = ap.parse_args()

    if not a.base_ckpt or not a.out:
        ap.error("--base-ckpt and --out are required")
    if a.w_group <= 0 or (a.a_group or a.w_group) <= 0:
        ap.error("group sizes must be positive")
    subspaces = None if str(a.subspaces).lower() in ("", "none") else a.subspaces
    export_checkpoint(a.base_ckpt, a.out, absmax=a.absmax, subspaces=subspaces, rank=a.rank,
                      smooth=a.smooth, rotate=a.rotate, w_bits=a.w_bits, a_bits=a.a_bits,
                      w_group=a.w_group, a_group=a.a_group or a.w_group, alpha=a.alpha,
                      fwht_block_max=a.fwht_block_max, verify=a.verify)


if __name__ == "__main__":
    main()
