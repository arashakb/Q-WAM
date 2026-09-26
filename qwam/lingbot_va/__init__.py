"""Q-WAM for LingBot-VA (shared video/action backbone, RoboTwin 2.0).

  qwam.lingbot_va.harness  in-process inference server, calibration frames, prompt cache
  qwam.lingbot_va.calib    activation absmax calibration for the smoothing factors, shard merge
  qwam.lingbot_va.aog      ASP subspaces from a sketched action observability Gramian
  qwam.lingbot_va.export   smoothing + block Hadamard + ASP + group-wise INT4 export
  qwam.lingbot_va.runtime  packed W4A4 Linears installed into the live server (LB_QUANT_CKPT)
"""
from .calib import calibrate_absmax, install_absmax_hooks, merge_shards
from .export import ExportConfig, export_checkpoint, quantize_linear, target_linears
from .runtime import W4A4PackedLinear, install_w4a4, load_checkpoint

__all__ = ["calibrate_absmax", "install_absmax_hooks", "merge_shards", "ExportConfig",
           "export_checkpoint", "quantize_linear", "target_linears", "W4A4PackedLinear",
           "install_w4a4", "load_checkpoint"]
