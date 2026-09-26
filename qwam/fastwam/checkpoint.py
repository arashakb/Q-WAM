"""Read the block linears of a Fast-WAM checkpoint without building the model."""
from __future__ import annotations

import re

import torch

# Checkpoint key of a block linear weight -> model module name (as named by target_linears).
_KEY = re.compile(r"mixtures\.(video|action)\.(blocks\.\d+\..+)\.weight")


def block_linears(ckpt_path) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """{module name: (weight [d_out, d_in], bias or None)} for the block linears of both experts.

    Tensors are memory-mapped, so reading only shapes costs no I/O beyond the file header.
    """
    obj = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    sd = obj["mot"] if isinstance(obj, dict) and "mot" in obj else obj
    out = {}
    for key, w in sd.items():
        m = _KEY.fullmatch(key)
        if m and torch.is_tensor(w) and w.ndim == 2:
            out[f"{m.group(1)}_expert.{m.group(2)}"] = (w, sd.get(key[:-len("weight")] + "bias"))
    if not out:
        raise ValueError(f"{ckpt_path}: no block linears found (expected 'mot' / mixtures.* keys)")
    return out
