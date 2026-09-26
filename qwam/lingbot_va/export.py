"""Q-WAM export for LingBot-VA: smoothing, block-Hadamard rotation, ASP and group-wise INT4.

Targets are the 300 Linears of the 30 transformer blocks (`blocks.N.*`: attn1/attn2 q, k, v, out and
the two FFN projections), 4,907,335,680 weights. LingBot-VA shares one backbone between video and
action, so every target layer gets an ASP branch.

Per target Linear y = x W^T (W is [out, in]), with s_j = a_j^alpha / w_j^(1-alpha) per input channel
(a: calibrated activation absmax, w: weight absmax) and H the block-diagonal orthonormal Hadamard of
block B = min(largest power of two dividing d_in, 1024):

    W_r = (W diag(s)) H                                  smoothed, rotated weight (fp32)
    ASP (orthonormal V [d_in, r] in the smoothed+rotated basis, from qwam.lingbot_va.aog):
        lr_a = V^T,  lr_b = W_r V,  W_r <- W_r (I - V V^T)
    qweight, wscale = symmetric INT4 codes of W_r, one scale per group of 32 input channels
    inv_smooth = 1/s (fp16)

qwam.lingbot_va.runtime implements the matching forward pass. The paper configuration is the
ExportConfig default with a rank-32 subspace file; Table 3 switches off ASP (no subspace file) and
additionally smoothing and rotation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from qwam.hadamard import block_fwht, hadamard_block
from qwam.quant import compute_smooth

TARGET_RE = re.compile(r"(^|\.)blocks\.\d+\.")
EXPECTED_LAYERS = 300

PACKING = ("qweight uint8, two int4 codes per byte, low nibble = even column, "
           "codes offset by +8 (stored 0..15, true range -8..7); wscale holds one scale per group "
           "of w_group input channels and is applied in fp16")
CALIBRATION = ("activation absmax over the RoboTwin 2.0 calibration set: 50 episodes "
               "(1 random per task, seed 42), all 10,919 frames")


@dataclass(frozen=True)
class ExportConfig:
    """Paper configuration: W4A4, group 32 for weights and activations, alpha 0.5, B <= 1024."""
    smooth: bool = True
    rotate: bool = True
    asp_rank: int = 32            # used only when a subspace file is given
    w_bits: int = 4
    a_bits: int = 4
    w_group: int = 32
    a_group: int = 32
    alpha: float = 0.5
    fwht_block_max: int = 1024

    def scheme(self, asp: bool) -> str:
        parts = [f"W{self.w_bits}A{self.a_bits}", "smooth" if self.smooth else "nosmooth"]
        if self.rotate:
            parts.append("hadamard")
        parts.append(f"g{self.w_group}")
        if asp:
            parts.append(f"ASP-r{self.asp_rank}")
        return "-".join(parts)


def target_linears(model: nn.Module) -> dict:
    """The block Linears of the LingBot-VA transformer, by module name."""
    return {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and TARGET_RE.search(n)}


def tensor_digest(tensors: dict) -> str:
    """sha256 over (name, dtype, shape, bytes) of a {name: tensor} dict; None values are skipped."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        v = tensors[k]
        if not torch.is_tensor(v):
            continue
        t = v.detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def group_quant_codes(w, bits, group):
    """Symmetric round-to-nearest codes of w [out, in] with one scale per group of `group` inputs."""
    out_f, in_f = w.shape
    assert in_f % group == 0, f"in_features {in_f} not divisible by group {group}"
    wg = w.reshape(out_f, in_f // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    codes = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax)
    return codes.reshape(out_f, in_f).to(torch.int8), scale.squeeze(-1)


def pack_int4(codes):
    """Two int4 codes per byte along the input axis: low nibble = even column, stored as code + 8."""
    c = (codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()


def unpack_int4(packed, in_features):
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(packed.shape[0], in_features, dtype=torch.int16)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequant_weight(e: dict, w_group: int) -> torch.Tensor:
    """fp32 reconstruction of the INT4 part of an entry: the stored W_r (deflated for ASP layers)."""
    inn, out = int(e["in_features"]), int(e["out_features"])
    q = unpack_int4(e["qweight"], inn).float()
    return (q.reshape(out, inn // w_group, w_group)
            * e["wscale"].float().unsqueeze(-1)).reshape(out, inn)


def quantize_linear(weight, bias, act_absmax, cfg: ExportConfig, V=None) -> dict:
    """Export entry of one Linear.

    weight [out, in] and bias in the model dtype; act_absmax [in] (ignored when cfg.smooth is off);
    V [in, >= cfg.asp_rank] orthonormal ASP basis in the smoothed+rotated coordinates, or None for
    a rank-0 layer. All math runs in fp32 on the CPU.
    """
    W = weight.float().cpu()                                        # [out, in]
    out_f, in_f = W.shape
    aa = act_absmax.float().cpu() if torch.is_tensor(act_absmax) else None
    s = compute_smooth(aa if cfg.smooth else None, W, cfg.alpha)   # [in]; ones without smoothing
    Wh = W * s.unsqueeze(0)
    B = hadamard_block(in_f, cfg.fwht_block_max) if cfg.rotate else 0
    if cfg.rotate and B < 4:
        raise ValueError(f"in_features {in_f} admits no Hadamard block >= 4")
    Wr = block_fwht(Wh, B) if cfg.rotate else Wh                    # (W diag(s)) H
    ent = {"inv_smooth": (1.0 / s.clamp(min=1e-8)).half(),
           "fwht_block": int(B),
           "in_features": int(in_f), "out_features": int(out_f)}
    if V is not None:
        V = V.float()[:, :cfg.asp_rank]                              # [in, r]
        if V.shape[0] != in_f:
            raise ValueError(f"V has {V.shape[0]} rows but in_features is {in_f}")
        lr_b = Wr @ V                                                # [out, r] = W_r V
        Wr = Wr - lr_b @ V.t()                                       # W_r (I - V V^T)
        ent["lr_a"] = V.t().contiguous().half()                      # [r, in] = V^T
        ent["lr_b"] = lr_b.contiguous().half()
        ent["asp_rank"] = int(cfg.asp_rank)
    codes, wscale = group_quant_codes(Wr, cfg.w_bits, cfg.w_group)
    ent["qweight"] = pack_int4(codes)
    ent["wscale"] = wscale                                           # fp32 here, fp16 at runtime
    if bias is not None:
        ent["bias"] = bias.detach().half().cpu()
    return ent


def check_subspaces(blob: dict, names, cfg: ExportConfig, absmax_digest: str | None) -> tuple[dict, list]:
    """ASP bases {layer: V} after checking coverage and that the basis matches the export settings.

    Returns (subspaces, warnings). Provenance fields absent from older files are reported, not
    enforced.
    """
    subs, smeta = blob["subspaces"], blob["meta"]
    warnings = []
    if not (cfg.smooth and cfg.rotate):
        raise ValueError("ASP needs smoothing and rotation: the subspace basis is smoothed+rotated")
    if smeta.get("basis") != "smoothed+rotated":
        raise ValueError(f"V must be in the smoothed+rotated basis, got {smeta.get('basis')!r}")
    rep, nfiles = smeta.get("episodes_represented"), smeta.get("episode_files")
    if rep is None or nfiles is None:
        raise ValueError("subspace meta lacks episodes_represented/episode_files")
    if rep < 0.9 * nfiles:
        raise ValueError(f"only {rep}/{nfiles} calibration episodes contributed to the subspaces")
    miss = [n for n in names if n not in subs]
    if miss:
        raise ValueError(f"subspaces miss {len(miss)}/{len(names)} target layers, e.g. {miss[:3]}")
    have = min(int(subs[n]["V"].shape[1]) for n in names)
    if cfg.asp_rank > have:
        raise ValueError(f"subspace file holds rank {have}, requested rank {cfg.asp_rank}")
    checks = (("alpha", cfg.alpha), ("fwht_block_max", cfg.fwht_block_max),
              ("absmax_digest", absmax_digest))
    for key, want in checks:
        if key not in smeta:
            warnings.append(f"subspace meta has no {key!r}; cannot confirm it matches the export")
        elif key == "alpha":
            if abs(float(smeta[key]) - float(want)) > 1e-12:
                raise ValueError(f"subspaces built at alpha={smeta[key]}, export uses {want}")
        elif smeta[key] != want:
            raise ValueError(f"subspaces built with {key}={smeta[key]}, export uses {want}")
    return subs, warnings


def bits_per_weight(payload: dict, cfg: ExportConfig) -> dict:
    """Weight bits: codes + one 16-bit scale per group, plus r (in + out) 16-bit values per ASP layer."""
    nparams = sum(e["in_features"] * e["out_features"] for e in payload.values())
    wbits = sum(e["in_features"] * e["out_features"] * cfg.w_bits
                + (e["in_features"] // cfg.w_group) * e["out_features"] * 16
                for e in payload.values())
    lr_bits = sum(int(e["asp_rank"]) * (e["in_features"] + e["out_features"]) * 16
                  for e in payload.values() if int(e.get("asp_rank", 0)) > 0)
    return {"quantized_params": nparams,
            "bpw_weights_only": round(wbits / nparams, 4),
            "bpw_with_asp_branch": round((wbits + lr_bits) / nparams, 4),
            "asp_branch_overhead_bits": lr_bits}


def build_meta(payload: dict, cfg: ExportConfig, asp: bool, provenance: dict) -> dict:
    rot = "(W diag(s)) H" if cfg.rotate else "W diag(s)"
    if asp:
        rot += " (I - V V^T)"
    act = ("bfwht(x * inv_smooth, fwht_block)" if cfg.rotate else "x * inv_smooth")
    meta = {"scheme": cfg.scheme(asp),
            "w_bits": cfg.w_bits, "a_bits": cfg.a_bits,
            "w_group": cfg.w_group, "a_group": cfg.a_group,
            "rank": cfg.asp_rank if asp else 0, "alpha": cfg.alpha,
            "smooth": cfg.smooth, "rotate": cfg.rotate,
            "requires_runtime_fwht": cfg.rotate, "fwht_block_max": cfg.fwht_block_max,
            "asp_ortho": asp,
            **bits_per_weight(payload, cfg),
            "n_layers": len(payload),
            "packing": PACKING,
            "dequant": (f"unpack(qweight) * wscale = {rot}"
                        + ("" if cfg.smooth else ", s = 1")
                        + f"; the runtime input to the INT4 GEMM is {act}"
                        + (" projected onto the complement of span(V)" if asp else "")),
            "runtime_contract": (
                "x_r = " + act + ("; p = x_r lr_a^T; y = Q_a(x_r - p lr_a) W_int4^T + p lr_b^T"
                                  if asp else "; y = Q_a(x_r) W_int4^T") + " + bias"),
            "calibration": CALIBRATION if (cfg.smooth or asp) else "none (smoothing off)",
            "base_checkpoint": "robbyant/lingbot-va-posttrain-robotwin",
            "targets": r"block Linears matched by (^|\.)blocks\.\d+\."}
    meta.update(provenance)
    return meta


def verify_payload(payload: dict, weights: dict, cfg: ExportConfig, subs: dict | None,
                   act_absmax: dict | None, n_layers: int = 3, log=print):
    """Self-checks on the first `n_layers` layers.

    1. relative error of the stored form against x W^T (INT4 weight noise only, a few percent;
       a value near 1 means the stored form does not match the runtime contract);
    2. for ASP, the float split x_r W_r^T = (I - VV^T)x_r (W_r(I - VV^T))^T + (V^T x_r)(W_r V)^T
       must hold to float precision.
    """
    gen = torch.Generator().manual_seed(0)
    names = list(payload)[:n_layers]
    for n in names:
        e = payload[n]
        Wq = dequant_weight(e, cfg.w_group)
        x = torch.randn(4, e["in_features"], generator=gen)
        ref = x @ weights[n].float().cpu().t()
        x_r = x * e["inv_smooth"].float()
        if int(e["fwht_block"]) >= 4:
            x_r = block_fwht(x_r, int(e["fwht_block"]))
        if "lr_a" in e:
            V = e["lr_a"].float().t()
            proj = x_r @ V
            got = (x_r - proj @ V.t()) @ Wq.t() + proj @ e["lr_b"].float().t()
        else:
            got = x_r @ Wq.t()
        rel = float((got - ref).norm() / (ref.norm() + 1e-12))
        log(f"[export] check {n[-46:]:46s} rel err {rel:.4f}")
        if rel > 0.5:
            raise RuntimeError(f"{n}: stored form does not reproduce the layer (rel err {rel:.3f})")
    if subs is None:
        return
    for n in names:
        W = weights[n].float().cpu()
        aa = act_absmax.get(n) if act_absmax is not None else None
        s = compute_smooth(aa.float().cpu() if torch.is_tensor(aa) else None, W, cfg.alpha)
        Wr = block_fwht(W * s.unsqueeze(0), hadamard_block(W.shape[1], cfg.fwht_block_max))
        V = subs[n]["V"].float()[:, :cfg.asp_rank]
        x_r = torch.randn(4, W.shape[1], generator=gen)
        lhs = x_r @ Wr.t()
        proj = x_r @ V
        rhs = (x_r - proj @ V.t()) @ (Wr - (Wr @ V) @ V.t()).t() + proj @ (Wr @ V).t()
        rel = float((rhs - lhs).norm() / (lhs.norm() + 1e-12))
        log(f"[export] ASP split {n[-46:]:46s} rel err {rel:.2e}")
        if rel >= 1e-4:
            raise RuntimeError(f"{n}: the ASP split is not an identity (rel err {rel:.2e})")


def export_checkpoint(model: nn.Module, cfg: ExportConfig, act_absmax: dict | None = None,
                      subspaces: dict | None = None, provenance: dict | None = None,
                      expected_layers: int | None = EXPECTED_LAYERS, verify: bool = True,
                      log=print) -> dict:
    """Quantize the block Linears of `model`; returns {"meta": ..., "layers": ...}.

    act_absmax: {layer: [in]} (required with smoothing); subspaces: a subspace file loaded with
    torch.load (enables ASP at rank cfg.asp_rank); provenance: extra meta fields.
    """
    targets = target_linears(model)
    names = list(targets)
    log(f"[export] {len(names)} target Linears, "
        f"{sum(m.weight.numel() for m in targets.values()) / 1e9:.2f} B params")
    if expected_layers is not None and len(names) != expected_layers:
        raise RuntimeError(f"expected {expected_layers} target Linears, found {len(names)}")
    for n, m in targets.items():
        if type(m) is not nn.Linear:
            raise RuntimeError(f"{n} is {type(m).__name__}, not nn.Linear (model already quantized?)")
    if cfg.smooth:
        if act_absmax is None:
            raise ValueError("smoothing needs the calibrated activation absmax")
        miss = [n for n in names if not torch.is_tensor(act_absmax.get(n))]
        if miss:
            raise ValueError(f"absmax lacks {len(miss)}/{len(names)} target layers, e.g. {miss[:3]}")
    absmax_digest = tensor_digest(act_absmax) if (cfg.smooth and act_absmax is not None) else None

    subs = None
    if subspaces is not None and cfg.asp_rank > 0:
        subs, warnings = check_subspaces(subspaces, names, cfg, absmax_digest)
        for w in warnings:
            log(f"[export] WARNING: {w}")
        sm = subspaces["meta"]
        log(f"[export] ASP rank {cfg.asp_rank} on all layers ({sm.get('episodes_represented')}/"
            f"{sm.get('episode_files')} episodes, {sm.get('frames')} frames)")

    payload = {}
    for j, n in enumerate(names):
        m = targets[n]
        payload[n] = quantize_linear(m.weight.data, m.bias.data if m.bias is not None else None,
                                     act_absmax.get(n) if act_absmax is not None else None, cfg,
                                     V=subs[n]["V"] if subs is not None else None)
        if (j + 1) % 50 == 0:
            log(f"[export] {j + 1}/{len(names)}")
    prov = {"absmax_digest": absmax_digest}
    prov.update(provenance or {})
    meta = build_meta(payload, cfg, subs is not None, prov)
    if verify:
        verify_payload(payload, {n: targets[n].weight.data for n in names}, cfg, subs,
                       act_absmax, log=log)
    log(f"[export] {meta['scheme']}: {meta['bpw_weights_only']:.4f} BPW weights only, "
        f"{meta['bpw_with_asp_branch']:.4f} BPW with the ASP branch")
    return {"meta": meta, "layers": payload}
