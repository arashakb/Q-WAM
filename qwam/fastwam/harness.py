"""Load Fast-WAM with its released RoboTwin checkpoint outside the simulator (for calibration).

The model is built exactly as the RoboTwin policy builds it (configs/sim_robotwin.yaml, text encoder
loaded, bf16), from the FastWAM checkout at $FASTWAM_ROOT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

RELEASED_CKPT = "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"

# Fast-WAM's RoboTwin inference settings (configs/sim_robotwin.yaml and the policy defaults).
ACTION_HORIZON = 32
NUM_INFERENCE_STEPS = 10


def fastwam_root() -> Path:
    root = os.environ.get("FASTWAM_ROOT")
    if not root:
        raise EnvironmentError("Set FASTWAM_ROOT to the FastWAM checkout.")
    return Path(root).resolve()


def load_model(device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16, ckpt: str | None = None):
    root = fastwam_root()
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Wan2.2 components are fetched into this directory on first use (FastWAM's README).
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="sim_robotwin.yaml")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt or root / RELEASED_CKPT))
    return model.to(device).eval()
