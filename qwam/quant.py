"""Symmetric group-wise fake quantization and the SmoothQuant smoothing factor."""
from __future__ import annotations

import torch


def group_quant_dequant(w: torch.Tensor, bits: int, group: int | None, dim: int = -1) -> torch.Tensor:
    """Symmetric round-to-nearest fake quantization of `w` along `dim`, one scale per group.

    The scale of a group is its absolute maximum divided by 2^(bits-1)-1, and codes are clamped
    to [-2^(bits-1), 2^(bits-1)-1]. If `group` is None or does not divide the dimension, one scale
    covers the whole dimension. bits <= 0 returns `w` unchanged.
    """
    if bits is None or bits <= 0:
        return w
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    if dim < 0:
        dim += wf.ndim
    n = wf.shape[dim]
    if group is None or n % group != 0:
        scale = wf.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wf / scale), -qmax - 1, qmax) * scale
        return q.to(w.dtype)
    shp = list(wf.shape)
    grouped = shp[:dim] + [n // group, group] + shp[dim + 1:]
    wr = wf.reshape(grouped)
    scale = wr.abs().amax(dim=dim + 1, keepdim=True).clamp(min=1e-8) / qmax
    q = (torch.clamp(torch.round(wr / scale), -qmax - 1, qmax) * scale).reshape(shp)
    return q.to(w.dtype)


def compute_smooth(act_absmax: torch.Tensor | None, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-input-channel SmoothQuant factor s_j = a_j^alpha / w_j^(1-alpha).

    `weight` is [out, in]; `act_absmax` is the per-input-channel absolute maximum of the layer
    input over the calibration set. The activation is divided by s and the weight multiplied by s.
    act_absmax=None disables smoothing (s = 1).
    """
    in_f = weight.shape[1]
    w_absmax = weight.float().abs().amax(dim=0).clamp(min=1e-5)  # [in]
    if act_absmax is None:
        return torch.ones(in_f, device=weight.device, dtype=torch.float32)
    a = act_absmax.float().to(weight.device).clamp(min=1e-5)
    s = a.pow(alpha) / w_absmax.pow(1.0 - alpha)
    return s.clamp(min=1e-5)
