"""Q-WAM for ImageWAM (FLUX.2 klein 4B editing expert + flow-matching action expert, RoboTwin 2.0).

  qwam.imagewam.calib    activation absmax hooks for the smoothing calibration, shard merge
  qwam.imagewam.export   smoothing + block Hadamard + ASP + group-wise INT4 export
  qwam.imagewam.runtime  W4A4 modules installed into a live ImageWAM policy
"""
from .calib import install_absmax_hooks, merge_shards
from .export import export_checkpoint, quantize_linear
from .runtime import W4A4PackedLinear, install_w4a4

__all__ = ["install_absmax_hooks", "merge_shards", "export_checkpoint", "quantize_linear",
           "W4A4PackedLinear", "install_w4a4"]
