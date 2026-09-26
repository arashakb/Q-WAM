"""LingBot-VA harness shared by calibration, AOG estimation and export.

Needs the patched lingbot-va checkout at $LINGBOT_ROOT (scripts/lingbot_va/setup_lingbot.sh). The
inference server is built in-process without torch.distributed, so the transformer is not FSDP
sharded and forward hooks see the plain nn.Linear modules.

Calibration frames are the raw per-camera dumps of scripts/common/dump_c50_frames.py (one
ep<episode>.npz per calibration episode, 320x240 uint8 per camera). They are fed to the server the
way the RoboTwin client feeds observations: the three camera images at 320x224, the instruction
through _reset.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
from pathlib import Path

import torch

OBS_KEYS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist")


def lingbot_root() -> Path:
    root = os.environ.get("LINGBOT_ROOT")
    if not root:
        raise EnvironmentError("Set LINGBOT_ROOT to the patched lingbot-va checkout.")
    path = Path(root).resolve()
    if not (path / "wan_va" / "wan_va_server.py").is_file():
        raise FileNotFoundError(f"{path} is not a lingbot-va checkout (no wan_va/wan_va_server.py)")
    return path


def _import_path(root: Path):
    for p in (str(root), str(root / "wan_va")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_server(config_name: str = "robotwin_i2av", enable_offload: bool = False,
                 save_root: str | None = None):
    """(VA_Server, config) on cuda:0, without distributed initialization."""
    root = lingbot_root()
    _import_path(root)
    os.chdir(root)                       # the i2av config reads its example images relative to it
    from configs import VA_CONFIGS
    from wan_va_server import VA_Server

    config = VA_CONFIGS[config_name]
    config.rank = 0
    config.local_rank = 0
    config.world_size = 0
    # _reset/_infer write per-frame debug tensors under save_root; keep them out of the checkout.
    config.save_root = save_root or os.path.join(tempfile.gettempdir(), "qwam_lingbot_va")
    os.makedirs(config.save_root, exist_ok=True)
    config.enable_offload = enable_offload
    print(f"[harness] building VA_Server({config_name!r}, offload={enable_offload})", flush=True)
    return VA_Server(config), config


def checkpoint_dir(config_name: str = "robotwin") -> str:
    """Base model directory as configured in the patched upstream config (LINGBOT_CKPT)."""
    _import_path(lingbot_root())
    from configs import VA_CONFIGS
    return VA_CONFIGS[config_name].wan22_pretrained_model_name_or_path


def load_transformer(ckpt_dir: str | None = None, device: str = "cpu", config_name: str = "robotwin"):
    """The LingBot-VA transformer alone, loaded as the server loads it (same dtype, for export)."""
    _import_path(lingbot_root())
    from configs import VA_CONFIGS
    from modules.utils import load_transformer as _load

    config = VA_CONFIGS[config_name]
    ckpt_dir = ckpt_dir or config.wan22_pretrained_model_name_or_path
    model = _load(os.path.join(ckpt_dir, "transformer"), torch_dtype=config.param_dtype,
                  torch_device=device, attn_mode="torch")
    return model.eval().requires_grad_(False)


def calib_files(raw_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no calibration dumps (*.npz) under {raw_dir}; "
                                f"run scripts/common/dump_c50_frames.py first")
    return files


def build_obs(ob: dict) -> dict:
    import cv2

    r = lambda a: cv2.resize(a, (320, 224))                                   # noqa: E731
    return {"obs": [{OBS_KEYS[0]: r(ob["head"]), OBS_KEYS[1]: r(ob["left"]),
                     OBS_KEYS[2]: r(ob["right"])}]}


class PromptCache:
    """Resets the server per frame and encodes each instruction once.

    _reset(prompt) resets the frame state and runs the text encoder; _reset(None) resets the state
    only. Restoring the cached embeddings after _reset(None) is therefore identical to re-encoding
    the same instruction. With park=True the text encoder stays on the CPU and is moved to the GPU
    only for an encode.
    """

    def __init__(self, server, park: bool = True):
        self.server = server
        self.park = park and getattr(server, "text_encoder", None) is not None
        self.device = next(server.transformer.parameters()).device
        self.prompt, self.embeds = None, None
        if self.park:
            server.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def invalidate(self):
        self.embeds = None

    def reset(self, prompt: str, force: bool = False):
        s = self.server
        if force or prompt != self.prompt or self.embeds is None:
            if self.park:
                s.text_encoder.to(self.device)
            s._reset(prompt)
            if self.park:
                s.text_encoder.to("cpu")
                torch.cuda.empty_cache()
            self.prompt = prompt
            self.embeds = (s.prompt_embeds, s.negative_prompt_embeds)
        else:
            s._reset(None)
            s.prompt_embeds, s.negative_prompt_embeds = self.embeds
