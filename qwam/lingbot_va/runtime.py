"""Packed W4A4 runtime for LingBot-VA checkpoints written by qwam.lingbot_va.export.

Every exported block Linear is replaced by a W4A4PackedLinear that keeps the packed INT4 weight
resident and dequantizes it per forward. With the tensors stored by the exporter:

    x_r = bfwht(x * inv_smooth, fwht_block)             rotation in fp32, result in the model dtype
    ASP layer:  p = x_r lr_a^T                          16-bit branch, never quantized
                y = Q_a(x_r - p lr_a) Q_w^T + p lr_b^T
    rank 0:     y = Q_a(x_r) Q_w^T
    y = y + bias

Q_w = int4 code * fp16 group scale (cast to the model dtype) holds the smoothed, rotated and, for
ASP layers, deflated weight W_r (I - V V^T); lr_a = V^T and lr_b = W_r V. Q_a is symmetric per-token
activation quantization with one scale per group of `a_group` channels. fwht_block = 0 means the
checkpoint was exported without rotation. inv_smooth, lr_a, lr_b and bias run in the dtype of the
Linear they replace (bf16 for LingBot-VA). This is simulated quantization: the GEMM runs in the
model dtype on the dequantized operands.

The upstream server installs a checkpoint when LB_QUANT_CKPT is set (patches/lingbot_va).
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwam.hadamard import block_fwht

# Entry fields this runtime implements; any other field belongs to a different export format.
ENTRY_KEYS = {"qweight", "wscale", "inv_smooth", "bias", "fwht_block", "in_features",
              "out_features", "asp_rank", "lr_a", "lr_b"}


class W4A4PackedLinear(nn.Module):
    """INT4 weight (packed, fp16 group scales), A4 activations, runtime FWHT, optional ASP branch."""

    def __init__(self, e: dict, group: int, a_bits: int, a_group: int, dtype, device):
        super().__init__()
        self.in_features = int(e["in_features"])
        self.out_features = int(e["out_features"])
        self.group = int(group)
        self.a_bits, self.a_group = a_bits, a_group
        self.register_buffer("qweight", e["qweight"].to(device))                 # uint8 [out, in/2]
        self.register_buffer("wscale", e["wscale"].to(device=device, dtype=torch.float16))
        self.register_buffer("inv_smooth", e["inv_smooth"].to(device=device, dtype=dtype))
        self.asp_rank = int(e.get("asp_rank", 0))
        self.register_buffer("lr_a", e["lr_a"].to(device=device, dtype=dtype)    # [r, in] = V^T
                             if self.asp_rank else None)
        self.register_buffer("lr_b", e["lr_b"].to(device=device, dtype=dtype)    # [out, r] = W_r V
                             if self.asp_rank else None)
        self.register_buffer("bias", e["bias"].to(device=device, dtype=dtype) if "bias" in e else None)
        self.fwht_block = int(e.get("fwht_block", 0))

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, w_group={self.group}, "
                f"a_bits={self.a_bits}, a_group={self.a_group}, fwht_block={self.fwht_block}, "
                f"asp_rank={self.asp_rank}")

    def _dequant_int4(self, dtype):
        q = self.qweight
        lo = (q & 0x0F).to(torch.int16) - 8
        hi = ((q >> 4) & 0x0F).to(torch.int16) - 8
        codes = torch.empty(self.out_features, self.in_features, dtype=torch.int16, device=q.device)
        codes[:, 0::2] = lo
        codes[:, 1::2] = hi
        W = codes.to(torch.float16).reshape(self.out_features, self.in_features // self.group,
                                            self.group)
        W = (W * self.wscale.unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return W.to(dtype)

    def _quant_act(self, x):
        if not self.a_bits or x.shape[-1] % self.a_group:
            return x
        qmax = 2 ** (self.a_bits - 1) - 1
        shp = x.shape
        xg = x.reshape(-1, shp[-1] // self.a_group, self.a_group).float()
        s = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        return (torch.clamp(torch.round(xg / s), -qmax - 1, qmax) * s).reshape(shp).to(x.dtype)

    def forward(self, x):
        xh = x * self.inv_smooth
        if self.fwht_block:                    # after smoothing, before activation quantization
            xh = block_fwht(xh.float(), self.fwht_block).to(x.dtype)
        if self.asp_rank:
            # Project first: the protected component p never reaches the activation quantizer.
            proj = xh @ self.lr_a.t()          # [..., r]  V^T x_r
            xh = xh - proj @ self.lr_a         # (I - V V^T) x_r
            xq = self._quant_act(xh)
            y = F.linear(xq, self._dequant_int4(xq.dtype))
            y = y + proj @ self.lr_b.t()
        else:
            xq = self._quant_act(xh)
            y = F.linear(xq, self._dequant_int4(xq.dtype))
        if self.bias is not None:
            y = y + self.bias
        return y


def load_checkpoint(ckpt_path) -> tuple[dict, dict]:
    """(meta, layers) of an exported checkpoint, after checking it is in the format run here."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"no quantized checkpoint at {ckpt_path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta, layers = blob["meta"], blob["layers"]
    if int(meta.get("w_bits", 4)) != 4:
        raise ValueError(f"{ckpt_path}: w_bits={meta['w_bits']}, the packed format holds 4-bit codes")
    group, a_group = int(meta["w_group"]), int(meta["a_group"])
    if group <= 0 or a_group <= 0:
        raise ValueError(f"{ckpt_path}: w_group={group}, a_group={a_group}; both must be positive")
    for name, e in layers.items():
        extra = set(e) - ENTRY_KEYS
        if extra:
            raise ValueError(f"{ckpt_path}: layer {name} carries unsupported fields {sorted(extra)}")
        if ("lr_a" in e) != (int(e.get("asp_rank", 0)) > 0):
            raise ValueError(f"{ckpt_path}: layer {name} has lr_a without asp_rank or vice versa")
        inn = int(e["in_features"])
        if inn % group or inn % a_group:
            raise ValueError(f"{ckpt_path}: layer {name} in_features={inn} is not a multiple of "
                             f"w_group={group} and a_group={a_group}")
    n_asp = sum(1 for e in layers.values() if int(e.get("asp_rank", 0)) > 0)
    if (meta.get("asp_ortho") or n_asp) and n_asp != len(layers):
        raise ValueError(f"{ckpt_path}: asp_ortho={meta.get('asp_ortho')} but {n_asp}/{len(layers)} "
                         f"layers carry an ASP branch")
    n_rot = sum(1 for e in layers.values() if int(e.get("fwht_block", 0)) > 0)
    if n_rot != (len(layers) if meta.get("requires_runtime_fwht") else 0):
        raise ValueError(f"{ckpt_path}: requires_runtime_fwht={meta.get('requires_runtime_fwht')} "
                         f"but {n_rot}/{len(layers)} layers carry fwht_block")
    return meta, layers


def install_w4a4(transformer: nn.Module, ckpt_path, verbose: bool = True) -> int:
    """Replace every Linear stored in the checkpoint; returns the number replaced (all or raise)."""
    meta, layers = load_checkpoint(ckpt_path)
    group, a_bits, a_group = meta["w_group"], meta["a_bits"], meta["a_group"]
    n_asp = sum(1 for e in layers.values() if int(e.get("asp_rank", 0)) > 0)
    n_rot = sum(1 for e in layers.values() if int(e.get("fwht_block", 0)) > 0)
    if verbose:
        print(f"[qwam] installing {meta.get('scheme')} from {ckpt_path} ({len(layers)} layers, "
              f"w_group={group}, a_bits={a_bits}, a_group={a_group})", flush=True)
        if n_asp:
            print(f"[qwam] ASP: rank {next(iter(layers.values()))['asp_rank']} protected subspace "
                  f"on all {n_asp} layers", flush=True)
        if n_rot:
            print(f"[qwam] rotated checkpoint: runtime FWHT enabled on all {n_rot} layers", flush=True)

    named = dict(transformer.named_modules())
    bad = [n for n, e in layers.items()
           if not isinstance(named.get(n), nn.Linear)
           or (named[n].in_features, named[n].out_features)
           != (int(e["in_features"]), int(e["out_features"]))]
    if bad:
        raise RuntimeError(f"{ckpt_path}: {len(bad)}/{len(layers)} layers have no matching nn.Linear "
                           f"in the model, e.g. {bad[:3]}")
    n_done = 0
    for name, e in layers.items():
        mod = named[name]
        new = W4A4PackedLinear(e, group, a_bits, a_group, mod.weight.dtype, mod.weight.device)
        parent = transformer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else transformer
        setattr(parent, name.rsplit(".", 1)[-1], new)
        named[name] = None                   # drop the last reference to the bf16 Linear
        del mod, new
        n_done += 1
        if n_done % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    n_total = len(layers)
    del named, layers
    gc.collect()
    msg = f"[qwam] replaced {n_done}/{n_total} Linears"
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        msg += f" | GPU {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB used after swap"
    if verbose:
        print(msg, flush=True)
    return n_done
