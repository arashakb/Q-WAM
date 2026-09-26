"""Q-WAM export for ImageWAM: smoothing, block-Hadamard rotation, ASP and group-wise INT4.

Targets are the MoT Linears that execute at inference: every `mixtures.action.*` Linear and every
`mixtures.video.transformer.*` Linear except the image-decoder head (`final_layer.linear`,
`final_layer.adaLN_modulation`), which never runs because the action expert reads the video KV
cache and no target frame is decoded. `mixtures.action.action_encoder` (14 -> 1024) has an input
width that is not a multiple of the weight group and stays in bf16. The released RoboTwin
checkpoint yields 153 quantized Linears (66 action expert, 87 video expert).

Per quantized Linear y = x W^T, with s_j = a_j^alpha / w_j^(1-alpha) per input channel (a: calibrated
activation absmax, w: weight absmax) and H the block-diagonal orthonormal Hadamard of block
B = min(largest power of two dividing d_in, 1024):

    W~ = (W diag(s)) H                                   smoothed, rotated weight (fp32)
    ASP layer (action expert, orthonormal V [d_in, r] in the smoothed+rotated basis):
        lr_a = V^T,  lr_b = W~ V,  W~ <- W~ (I - V V^T)
    qweight, wscale = symmetric INT4 codes of W~, one fp16 scale per group of input channels
    inv_smooth = 1/s (fp16)

qwam.imagewam.runtime implements the matching forward pass.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch

from qwam.hadamard import block_fwht, hadamard_block
from qwam.quant import compute_smooth

# Live MoT modules: the action expert and the video expert's `transformer` submodule.
LIVE = re.compile(r"^mixtures\.(action|video\.transformer)\.")
# Present on the model but never executed at inference.
DEAD = ("final_layer.linear", "final_layer.adaLN_modulation")
EXPECTED_LAYERS = 153

ASP_CONTRACT = ("x_r = bfwht(x*inv_smooth); proj = x_r @ lr_a^T; "
                "y = quant_act(x_r - proj @ lr_a) @ W_int4^T + proj @ lr_b^T; "
                "W_int4 holds W_r(I-VV^T) and proj is not quantized")


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


def select_targets(state_dict, w_group, rotate):
    """Sorted names of the Linears to quantize, and the live Linears kept in bf16."""
    names, kept_bf16 = [], []
    for k, v in state_dict.items():
        if not (k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2):
            continue
        n = k[: -len(".weight")]
        if not LIVE.match(n) or any(d in n for d in DEAD):
            continue
        if (v.shape[1] % w_group) or (rotate and v.shape[1] < 4):
            kept_bf16.append((n, int(v.shape[1]), int(v.shape[0]), int(v.numel())))
            continue
        names.append(n)
    names.sort()
    return names, kept_bf16


def quantize_linear(weight, act_absmax=None, *, V=None, rank=0, bias=None, w_bits=4, w_group=32,
                    alpha=0.5, rotate=True, fwht_block_max=1024):
    """Export entry of one Linear.

    weight [out, in]; act_absmax [in] or None (no smoothing, s = 1); V [in, >= rank] ASP basis in
    the smoothed+rotated coordinates, or None for a rank-0 layer. All math runs in fp32.
    """
    W = weight.float()                                               # [out, in]
    s = compute_smooth(act_absmax, W, alpha)                         # [in]
    Wh = W * s.unsqueeze(0)
    B = hadamard_block(W.shape[1], fwht_block_max) if rotate else 0
    Wr = block_fwht(Wh, B) if (rotate and B >= 4) else Wh            # (W diag(s)) H
    ent = {"inv_smooth": (1.0 / s.clamp(min=1e-8)).half(),
           "fwht_block": int(B),
           "in_features": int(W.shape[1]), "out_features": int(W.shape[0])}
    if V is not None:
        if V.shape[0] != W.shape[1]:
            raise ValueError(f"V has {V.shape[0]} rows but in_features is {W.shape[1]}")
        V = V.float()
        ent["lr_a"] = V.t().contiguous().half()                      # [r, in] = V^T
        ent["asp_rank"] = int(rank)
        lr_b = Wr @ V                                                # [out, r] = W~ V
        Wr = Wr - lr_b @ V.t()                                       # W~ (I - V V^T)
        ent["lr_b"] = lr_b.contiguous().half()
    codes, wscale = group_quant_codes(Wr, w_bits, w_group)
    ent["w_group"] = int(w_group)
    ent["qweight"] = pack_int4(codes)
    ent["wscale"] = wscale.half()
    if bias is not None:
        ent["bias"] = bias.half()
    return ent


def load_subspaces(path, rank, alpha):
    """ASP bases {layer: V [in, r] fp32} from a subspace file, after checking its provenance."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    subs, smeta = blob["subspaces"], blob["meta"]
    rep, nfiles = smeta.get("episodes_represented"), smeta.get("episode_files")
    if rep is None or nfiles is None:
        raise SystemExit(f"{path}: meta lacks episodes_represented/episode_files")
    if rep < 0.9 * nfiles:
        raise SystemExit(f"{path}: only {rep}/{nfiles} calibration episodes contributed")
    if smeta.get("basis") != "smoothed+rotated":
        raise SystemExit(f"{path}: V must be in the smoothed+rotated basis, got {smeta.get('basis')!r}")
    if smeta.get("expert") != "action":
        raise SystemExit(f"{path}: expected an action-expert subspace, got {smeta.get('expert')!r}")
    sub_alpha = float(smeta.get("alpha", 0.5))       # files without the field were built at 0.5
    if abs(sub_alpha - alpha) > 1e-9:
        raise SystemExit(f"{path} was built at alpha={sub_alpha}, export uses alpha={alpha}; "
                         f"the basis depends on the smoothing factor")
    have = min(int(e["V"].shape[1]) for e in subs.values())
    if rank > have:
        raise SystemExit(f"{path} holds rank {have}, requested rank {rank}")
    bases = {n: e["V"].float()[:, :rank] for n, e in subs.items()}
    return bases, smeta


def export_checkpoint(base_ckpt, out, absmax=None, subspaces=None, rank=32, smooth=True,
                      rotate=True, w_bits=4, a_bits=4, w_group=32, a_group=None, alpha=0.5,
                      fwht_block_max=1024, verify=True):
    """Quantize the released ImageWAM checkpoint and write {"meta", "layers"} to `out`."""
    a_group = a_group or w_group
    use_asp = bool(subspaces) and rank > 0
    if use_asp and not (smooth and rotate):
        raise SystemExit("ASP needs smoothing and rotation: the subspace basis is smoothed+rotated")
    if smooth and not absmax:
        raise SystemExit("smoothing needs the calibrated activation absmax (--absmax)")

    print(f"[export] base checkpoint {base_ckpt}", flush=True)
    mot = torch.load(base_ckpt, map_location="cpu", weights_only=False, mmap=True)["mot"]
    names, kept_bf16 = select_targets(mot, w_group, rotate)
    print(f"[export] {len(names)} target Linears, kept in bf16: "
          f"{', '.join(f'{n}({i}->{o})' for n, i, o, _ in kept_bf16) or 'none'}", flush=True)
    if len(names) != EXPECTED_LAYERS:
        print(f"[export] WARNING: expected {EXPECTED_LAYERS} target Linears for the released "
              f"RoboTwin checkpoint, found {len(names)}", flush=True)

    act, calib_note = {}, "none (smoothing off)"
    if smooth:
        blob = torch.load(absmax, map_location="cpu", weights_only=False)
        act = blob["absmax"] if isinstance(blob, dict) and "absmax" in blob else blob
        missing = [n for n in names if n not in act]
        if missing:
            raise SystemExit(f"absmax lacks {len(missing)}/{len(names)} target Linears, "
                             f"e.g. {missing[:3]}")
        ntask = len(blob.get("tasks", [])) if isinstance(blob, dict) else 0
        calib_note = f"activation absmax over {ntask} RoboTwin tasks, 1 closed-loop episode each"
        print(f"[export] absmax {absmax}: {len(act)} layers, {ntask} tasks", flush=True)

    bases, smeta = {}, None
    if use_asp:
        bases, smeta = load_subspaces(subspaces, rank, alpha)
        off_scope = [n for n in names if n in bases and not n.startswith("mixtures.action.")]
        if off_scope:
            raise SystemExit(f"{subspaces} has bases outside the action expert: {off_scope[:3]}")
        print(f"[export] ASP rank {rank} on {sum(n in bases for n in names)} action-expert Linears "
              f"({smeta.get('episodes_represented')}/{smeta.get('episode_files')} episodes, "
              f"{smeta.get('frames')} frames)", flush=True)

    payload, branch_bits, n_asp, t0 = {}, 0, 0, time.time()
    for j, n in enumerate(names):
        V = bases.get(n)
        ent = quantize_linear(mot[f"{n}.weight"], act[n].float() if smooth else None, V=V,
                              rank=rank, bias=mot.get(f"{n}.bias"), w_bits=w_bits,
                              w_group=w_group, alpha=alpha, rotate=rotate,
                              fwht_block_max=fwht_block_max)
        if V is not None:
            branch_bits += rank * (ent["out_features"] + ent["in_features"]) * 16
            n_asp += 1
        payload[n] = ent
        if (j + 1) % 40 == 0:
            print(f"[export] {j + 1}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)

    nparams = sum(e["in_features"] * e["out_features"] for e in payload.values())
    wbits = sum(e["in_features"] * e["out_features"] * w_bits
                + (e["in_features"] // e["w_group"]) * e["out_features"] * 16
                for e in payload.values())
    n_act = sum(e["in_features"] * e["out_features"] for k, e in payload.items()
                if k.startswith("mixtures.action."))
    base = ("hadamard" if rotate else "smooth") if smooth else ("rotonly" if rotate else "group")
    scheme = f"W{w_bits}A{a_bits}-{base}" + (f"-ASPortho-action-r{rank}" if use_asp else "")
    meta = {
        "scheme": scheme,
        "asp_ortho": use_asp, "asp_expert": "action" if use_asp else None,
        "asp_layers": n_asp, "rank": rank if use_asp else 0,
        "asp_subspaces": Path(subspaces).name if use_asp else None,
        "asp_branch_overhead_bits": branch_bits,
        "bpw_with_asp_branch": round((wbits + branch_bits) / nparams, 4),
        "bpw_weights_only": round(wbits / nparams, 4),
        "asp_runtime_contract": ASP_CONTRACT if use_asp else "n/a (rank 0)",
        "w_bits": w_bits, "a_bits": a_bits, "w_group": w_group, "a_group": a_group,
        "alpha": alpha, "smooth": smooth, "rotate": rotate, "requires_runtime_fwht": rotate,
        "fwht_block_max": fwht_block_max,
        "n_layers": len(payload), "quantised_params": nparams,
        "action_expert_params": n_act, "action_expert_share": round(n_act / nparams, 4),
        "excluded_dead_layers": [k for k in mot if k.endswith(".weight") and any(d in k for d in DEAD)],
        "kept_bf16_ungroupable": [{"name": n, "in": i, "out": o, "params": p}
                                  for n, i, o, p in kept_bf16],
        "packing": "qweight uint8, two int4 codes per byte, low nibble = even column, "
                   "codes offset by +8 (stored 0..15, true range -8..7)",
        "dequant": "W ~= unpack(qweight) * wscale is (W*s)H; the runtime applies "
                   "bfwht(x*inv_smooth, fwht_block) to the activation",
        "calibration": calib_note,
        "base_checkpoint": Path(base_ckpt).name,
        "targets": "live MoT Linears: mixtures.action.* + mixtures.video.transformer.*",
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "layers": payload}, out)
    meta_json = Path(out).with_name(Path(out).stem + "_meta.json")
    meta_json.write_text(json.dumps(meta, indent=2, default=str))
    print(f"[export] wrote {out} ({os.path.getsize(out) / 2**30:.2f} GiB, {len(payload)} layers, "
          f"{n_asp} with ASP, {meta['bpw_with_asp_branch']:.4f} BPW)", flush=True)
    if verify:
        verify_export(payload, mot)
    return meta


def dequant_weight(e):
    g = int(e["w_group"])
    Wq = unpack_int4(e["qweight"], e["in_features"]).float()
    return (Wq.reshape(e["out_features"], e["in_features"] // g, g)
            * e["wscale"].float().unsqueeze(-1)).reshape(e["out_features"], e["in_features"])


def verify_export(payload, mot, per_kind=3, seed=0):
    """Relative error of the stored form against x W^T on random inputs (weight rounding only).

    About 0.1-0.3 is INT4 weight noise on random inputs; above 0.5 means the stored form does not
    match the rotation/deflation contract.
    """
    gen = torch.Generator().manual_seed(seed)
    asp = [n for n in payload if "lr_a" in payload[n]][:per_kind]
    plain = [n for n in payload if "lr_a" not in payload[n]][:per_kind]
    for n in asp + plain:
        e = payload[n]
        Wq = dequant_weight(e)
        x = torch.randn(4, e["in_features"], generator=gen)
        ref = x @ mot[f"{n}.weight"].float().t()
        xr = x * e["inv_smooth"].float()
        if e["fwht_block"] >= 4:
            xr = block_fwht(xr, e["fwht_block"])
        if "lr_a" in e:
            V = e["lr_a"].float().t()
            proj = xr @ V
            got = (xr - proj @ V.t()) @ Wq.t() + proj @ e["lr_b"].float().t()
        else:
            got = xr @ Wq.t()
        rel = float((got - ref).norm() / (ref.norm() + 1e-12))
        print(f"[export] check {'ASP ' if 'lr_a' in e else 'rank0'} {n[-48:]:48s} "
              f"rel err {rel:.4f}", flush=True)
