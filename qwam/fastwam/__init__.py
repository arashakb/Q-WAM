"""Q-WAM for Fast-WAM (video-generation MoT: 5B video expert + 1B action expert, RoboTwin 2.0).

  qwam.fastwam.asp_linear   ASPLinear: simulated W4A4 layer with smoothing, rotation and ASP
  qwam.fastwam.install      QWAMConfig, target_linears, cache loaders, install_qwam, install_from_env
  qwam.fastwam.harness      loads the released RoboTwin checkpoint for calibration (needs FASTWAM_ROOT)
  qwam.fastwam.checkpoint   block-linear shapes and weights read directly from a checkpoint

The RoboTwin policy hook (patches/fastwam/deploy_policy.patch) calls install_from_env when
QWAM_ENABLE=1.
"""
from .asp_linear import ASPLinear
from .install import (QWAMConfig, install_from_env, install_qwam, load_absmax, load_aog,
                      target_linears)

__all__ = ["ASPLinear", "QWAMConfig", "install_from_env", "install_qwam", "load_absmax",
           "load_aog", "target_linears"]
