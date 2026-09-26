"""Merge the per-task activation absmax shards written by the calibration run.

  python scripts/imagewam/merge_absmax.py --dir <calib dir> --prefix absmax \
      --out <calib dir>/imagewam_act_absmax_c50.pt --expect-layers 154
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam.imagewam.calib import EXPECTED_LAYERS, merge_shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the shards")
    ap.add_argument("--prefix", default="absmax", help="shard prefix (IW_CALIB_OUT basename)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-layers", type=int, default=EXPECTED_LAYERS)
    a = ap.parse_args()
    merge_shards(a.dir, a.prefix, a.out, a.expect_layers)


if __name__ == "__main__":
    main()
