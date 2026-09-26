"""W4A4 runtime for ImageWAM: replaces the exported MoT Linears of a live policy.

Forward of one quantized Linear (see qwam.imagewam.export for the stored tensors):

    x~ = bfwht(x * inv_smooth)                          rotation in fp32, result in the model dtype
    ASP layer:   p = x~ lr_a^T                          16-bit branch, never quantized
                 y = Q_a(x~ - p lr_a) Q_w^T + p lr_b^T
    rank 0:      y = Q_a(x~) Q_w^T
    y = y + bias

Q_w is the dequantized INT4 weight (int4 code * fp16 group scale, cast to the model dtype) and Q_a is
symmetric per-token activation quantization with one scale per group of `a_group` channels. This is
simulated quantization: the GEMM runs in the model dtype on the dequantized operands.

Enable in the ImageWAM RoboTwin policy with IW_QUANT_CKPT=<exported checkpoint>.
IW_CACHE_WEIGHTS=0 disables the resident dequantized-weight cache (same outputs, less memory).
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn

from qwam.hadamard import block_fwht

# Entry fields this runtime implements; anything else belongs to a different export format.
ENTRY_KEYS = {"qweight", "wscale", "inv_smooth", "bias", "fwht_block", "in_features",
              "out_features", "w_group", "asp_rank", "lr_a", "lr_b"}


def _unpack_int4(packed: torch.Tensor, in_features: int, device) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(packed.shape[0], in_features, dtype=torch.int16, device=device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


class W4A4PackedLinear(nn.Module):
    """INT4 weight (packed), A4 activations, runtime block Hadamard, optional ASP branch."""

    def __init__(self, e: dict, group: int, a_bits: int, a_group: int, dtype, device):
        super().__init__()
        self.in_features = int(e["in_features"])
        self.out_features = int(e["out_features"])
        self.group = int(e.get("w_group", group))
        self.a_bits, self.a_group = a_bits, a_group
        self.register_buffer("qweight", e["qweight"].to(device))
        self.register_buffer("wscale", e["wscale"].to(device=device, dtype=torch.float16))
        self.register_buffer("inv_smooth", e["inv_smooth"].to(device=device, dtype=dtype))
        self.register_buffer("bias",
                             e["bias"].to(device=device, dtype=dtype) if "bias" in e else None)
        # ASP: lr_a = V^T [r, in], lr_b = W~V [out, r]; qweight holds the deflated W~(I - VV^T).
        self.asp_rank = int(e.get("asp_rank", 0))
        self.register_buffer("lr_a", e["lr_a"].to(device=device, dtype=dtype)
                             if "lr_a" in e else None)
        self.register_buffer("lr_b", e["lr_b"].to(device=device, dtype=dtype)
                             if "lr_b" in e else None)
        # 0: the stored weight is not rotated and the activation is not rotated either.
        self.fwht_block = int(e.get("fwht_block", 0))
        self._wcache = None
        self._want_cache = os.environ.get("IW_CACHE_WEIGHTS", "1") != "0"

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, w_group={self.group}, "
                f"a_bits={self.a_bits}, a_group={self.a_group}, fwht_block={self.fwht_block}, "
                f"asp_rank={self.asp_rank}")

    def _dequant_int4(self, dtype):
        codes = _unpack_int4(self.qweight, self.in_features, self.qweight.device)
        W = codes.to(torch.float16).reshape(
            self.out_features, self.in_features // self.group, self.group)
        W = (W * self.wscale.unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return W.to(dtype)

    def _quant_act(self, x, bits=None):
        bits = self.a_bits if bits is None else bits
        if not bits or x.shape[-1] % self.a_group:
            return x
        qmax = 2 ** (bits - 1) - 1
        shp = x.shape
        xg = x.reshape(-1, shp[-1] // self.a_group, self.a_group).float()
        s = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        return (torch.clamp(torch.round(xg / s), -qmax - 1, qmax) * s).reshape(shp).to(x.dtype)

    def _weight(self, dtype):
        """Dequantized weight; kept resident on the GPU while at least 6 GiB stay free."""
        if self._wcache is not None and self._wcache.dtype == dtype:
            return self._wcache
        W = self._dequant_int4(dtype)
        if self._want_cache and self._wcache is None and W.is_cuda:
            need = W.numel() * W.element_size()
            free, _ = torch.cuda.mem_get_info(W.device)
            if free - need > 6 * 2 ** 30:
                self._wcache = W
        return W

    def forward(self, x):
        xh = x * self.inv_smooth
        if self.fwht_block:
            # after smoothing, before activation quantization; the stored weight is (W*s)H
            xh = block_fwht(xh.float(), self.fwht_block).to(x.dtype)
        if self.asp_rank:
            proj = xh @ self.lr_a.t()                    # V^T x~
            xh = xh - proj @ self.lr_a                   # (I - VV^T) x~
            xq = self._quant_act(xh)
            W = self._weight(xq.dtype)
            y = torch.nn.functional.linear(xq, W)
            del W
            y = y + proj @ self.lr_b.t()
        else:
            xq = self._quant_act(xh)
            W = self._weight(xq.dtype)
            y = torch.nn.functional.linear(xq, W)
            del W
        if self.bias is not None:
            y = y + self.bias
        return y


def install_w4a4(model, ckpt_path=None, verbose=True) -> int:
    """Replace every exported Linear inside model.mot. Returns the number replaced."""
    ckpt_path = ckpt_path or os.environ.get("IW_QUANT_CKPT")
    if not ckpt_path or not Path(ckpt_path).exists():
        if verbose:
            print(f"[iw-quant] no checkpoint at {ckpt_path!r}; leaving model in bf16", flush=True)
        return 0

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta, layers = blob["meta"], blob["layers"]
    a_bits, a_group, group = meta["a_bits"], meta["a_group"], meta["w_group"]
    if verbose:
        print(f"[iw-quant] installing {meta['scheme']} from {ckpt_path} "
              f"({len(layers)} layers, w_group={group})", flush=True)

    unknown = sorted({k for e in layers.values() for k in e} - ENTRY_KEYS)
    if unknown:
        raise RuntimeError(f"{ckpt_path}: unsupported layer fields {unknown}; "
                           f"not a Q-WAM ImageWAM checkpoint")
    n_asp = sum(1 for e in layers.values() if int(e.get("asp_rank", 0)) > 0)
    if meta.get("asp_ortho") or n_asp:
        want = int(meta.get("asp_layers", -1))
        if want >= 0 and n_asp != want:
            raise RuntimeError(f"{ckpt_path}: meta says asp_layers={want} but {n_asp} layers "
                               f"carry asp_rank")
        scope = str(meta.get("asp_expert") or "action")
        off_scope = [k for k, e in layers.items()
                     if int(e.get("asp_rank", 0)) > 0 and not k.startswith("mixtures.action.")]
        if scope != "action" or off_scope:
            raise RuntimeError(f"{ckpt_path}: ASP is supported on the action expert only "
                               f"(asp_expert={scope!r}, e.g. {off_scope[:3]})")
        missing_branch = [k for k, e in layers.items()
                          if int(e.get("asp_rank", 0)) > 0 and not ("lr_a" in e and "lr_b" in e)]
        if missing_branch:
            raise RuntimeError(f"{ckpt_path}: ASP layers without lr_a/lr_b: {missing_branch[:3]}")
        if verbose:
            rank = next(int(e["asp_rank"]) for e in layers.values() if e.get("asp_rank"))
            print(f"[iw-quant] ASP on {n_asp} action-expert Linears (rank {rank})", flush=True)
    if meta.get("requires_runtime_fwht"):
        n_rot = sum(1 for e in layers.values() if int(e.get("fwht_block", 0)) > 0)
        if n_rot != len(layers):
            raise RuntimeError(f"{ckpt_path} is rotated but only {n_rot}/{len(layers)} layers "
                               f"carry fwht_block")
        if verbose:
            print(f"[iw-quant] rotated checkpoint: runtime FWHT on all {n_rot} layers", flush=True)

    # Export keys are module paths relative to the MoT.
    mot = getattr(model, "mot", model)
    named = dict(mot.named_modules())
    n_done, missing, wrong_type = 0, [], []
    for name, e in layers.items():
        mod = named.get(name)
        if mod is None:
            missing.append(name)
            continue
        if not isinstance(mod, nn.Linear):
            wrong_type.append(f"{name}({type(mod).__name__})")
            continue
        new = W4A4PackedLinear(e, group, a_bits, a_group, mod.weight.dtype, mod.weight.device)
        parent = mot.get_submodule(name.rsplit(".", 1)[0]) if "." in name else mot
        setattr(parent, name.rsplit(".", 1)[-1], new)
        named[name] = None            # release the bf16 weight
        del mod, new
        n_done += 1
        if n_done % 50 == 0:
            torch.cuda.empty_cache()

    n_total = len(layers)
    del named, blob, layers
    gc.collect()
    torch.cuda.empty_cache()
    # A partially quantized model is not a defined configuration.
    if missing or wrong_type:
        raise RuntimeError(f"{ckpt_path}: replaced only {n_done}/{n_total} Linears. "
                           f"missing={missing[:3]} wrong_type={wrong_type[:3]}")
    if verbose:
        msg = f"[iw-quant] replaced {n_done}/{n_total} Linears"
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            msg += f" | GPU {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB used after swap"
        print(msg, flush=True)
    return n_done
