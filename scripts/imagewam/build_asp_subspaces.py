r"""Action-output Gram (AOG) and per-layer ASP subspaces for ImageWAM's action expert (one GPU).

For every Linear l of the action expert, with x_l its input activation:

    G_l = E[J_l^T J_l],   J_l = d(action-expert output) / d(x_l)

The output is the action expert's prediction at the FIRST denoising step of the action sampler
(action_step_idx 0). The video expert is prefilled once into a KV cache before the action loop and
is a constant with respect to this gradient. Each frame contributes one Hutchinson probe (u^T J_l,
u ~ N(0, I) seeded by the frame) to a Nystrom sketch G_l Omega_l with rank + oversample columns.
The sketch is mapped to the smoothed+rotated basis the quantizer sees,
G~_l = H diag(s) G_l diag(s) H (s from the calibrated absmax at alpha 0.5, H the block Hadamard with
block cap 1024), and V_l is the top-`rank` eigenbasis of the Nystrom approximation of G~_l.
Layers whose input does not depend on the action-expert parameters (action_encoder,
time_in.in_layer) receive no gradient and are skipped.

Frames: `--per-ep` evenly spaced frames from each calibration episode dumped by
scripts/common/dump_c50_frames.py (default 2 per episode, 100 frames; 0 = all frames).

  set -a; source $IMAGEWAM_ROOT/.env.local; set +a
  python scripts/imagewam/build_asp_subspaces.py --frames-dir <c50 frames> \
      --out <imagewam_asp_subspaces_r32.pt>
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwam.hadamard import block_fwht, hadamard_block  # noqa: E402
from qwam.quant import compute_smooth  # noqa: E402

ALPHA = 0.5
FWHT_BLOCK_MAX = 1024
SIM_TASK = "robotwin_flux2_klein_4b_base_clean_imagewam"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--imagewam-root", default=os.environ.get("IMAGEWAM_ROOT"))
ap.add_argument("--frames-dir", default=os.environ.get("C50_RAW"),
                help="calibration frames (ep*.npz) from scripts/common/dump_c50_frames.py [C50_RAW]")
ap.add_argument("--absmax", default=os.environ.get("IW_ABSMAX", str(
    Path(__file__).resolve().parents[2] / "artifacts/imagewam/imagewam_act_absmax_c50.pt")),
                help="merged activation absmax; default: the shipped calibration [IW_ABSMAX]")
ap.add_argument("--ckpt", default=None, help="default: $CKPT_PATH or the released RoboTwin model.pt")
ap.add_argument("--dataset-stats", default=None, help="default: $DATASET_STATS_PATH or next to --ckpt")
ap.add_argument("--flux2-src", default=os.environ.get("FLUX2_SRC"))
ap.add_argument("--flux2-model", default=os.environ.get("FLUX2_MODEL_PATH"))
ap.add_argument("--flux2-ae", default=os.environ.get("FLUX2_AE_MODEL_PATH"))
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--oversample", type=int, default=32)
ap.add_argument("--per-ep", type=int, default=2, help="frames per episode; 0 = all frames")
ap.add_argument("--nprobe", type=int, default=1)
ap.add_argument("--seed", type=int, default=42, help="seeds the probes, the sketch and the sampler noise")
args = ap.parse_args()

for k in ("imagewam_root", "frames_dir", "absmax", "flux2_src", "flux2_model", "flux2_ae"):
    if not getattr(args, k):
        ap.error(f"--{k.replace('_', '-')} is required (or its environment variable)")
# the policy is built from the ImageWAM root, so every user path is made absolute first
for k in ("imagewam_root", "frames_dir", "absmax", "ckpt", "dataset_stats", "flux2_src",
          "flux2_model", "flux2_ae", "out"):
    if getattr(args, k):
        setattr(args, k, str(Path(getattr(args, k)).resolve()))
IW = args.imagewam_root
CKPT = str(Path(args.ckpt or os.environ.get("CKPT_PATH")
                or f"{IW}/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt").resolve())
STATS = str(Path(args.dataset_stats or os.environ.get("DATASET_STATS_PATH")
                 or Path(CKPT).with_name("dataset_stats.json")).resolve())

sys.path.insert(0, f"{IW}/src")
sys.path.insert(0, f"{IW}/experiments/robotwin")
sys.path.insert(0, f"{IW}/experiments/robotwin/imagewam_policy")
sys.path.insert(0, f"{args.flux2_src}/src")
os.chdir(IW)

Q = args.rank + args.oversample
_OM = {}


def omega(nm, d_in, device):
    """Sketch test matrix of one layer, seeded by sha1(name) so it is identical across processes."""
    key = (nm, d_in)
    if key not in _OM:
        h = int(hashlib.sha1(nm.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device="cpu").manual_seed((h ^ (args.seed * 7919)) % (2 ** 31))
        _OM[key] = torch.randn(d_in, Q, generator=g)
    return _OM[key].to(device)


def frames(files, per_ep):
    for ep, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        H, L, R, S = d["head"], d["left"], d["right"], d["state"]
        p = str(d["prompt"])
        idx = (np.arange(H.shape[0]) if per_ep <= 0
               else np.linspace(0, H.shape[0] - 1, num=min(per_ep, H.shape[0]), dtype=int))
        for i in idx:
            yield dict(head=H[i], left=L[i], right=R[i], state=S[i], prompt=p, ep=ep, fi=int(i))


def to_obs(fr):
    return {"observation": {"head_camera": {"rgb": fr["head"]},
                            "left_camera": {"rgb": fr["left"]},
                            "right_camera": {"rgb": fr["right"]}},
            "joint_action": {"vector": fr["state"]}}


def main():
    from deploy_policy import DEFAULT_PROMPT, get_model
    usr = {"sim_cfg_path": None, "sim_cfg_name": "sim_robotwin", "sim_task": SIM_TASK,
           "ckpt_setting": CKPT, "dataset_stats_path": STATS, "device": "cuda",
           "model_overrides": {"flux2_src_path": args.flux2_src,
                               "flux2_model_path": args.flux2_model,
                               "ae_model_path": args.flux2_ae}}
    os.environ["IW_QUANT_CKPT"] = ""           # the AOG is measured on the bf16 model
    os.environ.pop("IW_CALIB_OUT", None)
    pol = get_model(usr)
    mot = pol.model.mot

    TGT = {n: m for n, m in mot.named_modules()
           if isinstance(m, nn.Linear) and n.startswith("mixtures.action.")}
    print(f"[aog] {len(TGT)} action-expert Linears", flush=True)
    assert len(TGT) >= 60, f"expected ~67 action-expert Linears, got {len(TGT)}"

    # Captured activations are graph nodes only if the expert's parameters require grad.
    for p in pol.model.action_expert.parameters():
        p.requires_grad_(True)

    caught, stash = {}, {}

    def mk(name):
        def hook(mod, inp):
            x = inp[0]
            if (stash.get("arm") and torch.is_grad_enabled()
                    and torch.is_tensor(x) and x.requires_grad):
                x.retain_grad()
                caught[name] = x
        return hook

    for n, m in TGT.items():
        m.register_forward_pre_hook(mk(n))

    sketch = {n: torch.zeros(m.in_features, Q, dtype=torch.float32) for n, m in TGT.items()}
    trG = {n: 0.0 for n in TGT}

    # Differentiate the action expert's output head on its first call of an inference, i.e. at
    # the first denoising step; later steps run without a graph.
    orig_post = pol.model.action_expert.post_dit

    def post_dit(*a, **kw):
        if not stash.get("arm") or "done" in stash:
            return orig_post(*a, **kw)
        with torch.enable_grad():
            out = orig_post(*a, **kw)
            if torch.is_tensor(out) and out.requires_grad and caught:
                names = list(caught.keys())
                for p in range(args.nprobe):
                    # probe seeded by (episode, frame, probe, seed): the estimate depends only on
                    # the frame set
                    _k = f"{stash.get('ep')}:{stash.get('fi')}:{p}:{args.seed}"
                    _g = torch.Generator(device="cpu").manual_seed(
                        int(hashlib.sha1(_k.encode()).hexdigest()[:8], 16))
                    u = torch.randn(out.shape, generator=_g, dtype=torch.float32).to(
                        out.device, out.dtype)
                    gs = torch.autograd.grad(
                        (out * u).sum(), [caught[n] for n in names],
                        retain_graph=(p < args.nprobe - 1), allow_unused=True)
                    for nm, g in zip(names, gs):
                        if g is None:
                            continue
                        gm = g.detach().reshape(-1, g.shape[-1]).float()
                        sketch[nm] += (gm.t() @ (gm @ omega(nm, gm.shape[-1], gm.device))).cpu()
                        trG[nm] += float(gm.pow(2).sum())
                stash["done"] = True
            for t in caught.values():
                t.grad = None
            caught.clear()
            return out.detach() if torch.is_tensor(out) else out

    pol.model.action_expert.post_dit = post_dit

    # infer_action_flux2 is decorated with @torch.no_grad(); call the undecorated function.
    _raw = getattr(type(pol.model).infer_action_flux2, "__wrapped__", None)
    if _raw is None:
        raise SystemExit("infer_action_flux2 has no __wrapped__; cannot differentiate through it")

    # Math SDPA backend: the fused attention kernels do not support this backward at every shape.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        _sdpa = lambda: sdpa_kernel([SDPBackend.MATH])                    # noqa: E731
    except ImportError:
        _sdpa = lambda: torch.backends.cuda.sdp_kernel(                    # noqa: E731
            enable_flash=False, enable_mem_efficient=False, enable_math=True)

    def infer_with_grad(fr):
        img = pol._build_robotwin_image_tensor(to_obs(fr))
        proprio = pol._normalize_state(np.asarray(fr["state"], dtype=np.float32))
        # sampler noise seeded by the frame
        fseed = pol.seed if pol.seed is not None else int(
            hashlib.sha1(f"{fr['ep']}:{fr.get('fi', -1)}:{args.seed}".encode()
                         ).hexdigest()[:8], 16) % (2 ** 31)
        with _sdpa(), torch.enable_grad():
            return _raw(pol.model,
                        prompt=DEFAULT_PROMPT.format(task=fr["prompt"]),
                        input_image=img, action_horizon=pol.action_horizon, proprio=proprio,
                        num_inference_steps=pol.num_inference_steps,
                        sigma_shift=pol.sigma_shift, seed=fseed,
                        rand_device=pol.rand_device)

    files = sorted(glob.glob(f"{args.frames_dir}/*.npz"))
    assert files, f"no calibration frames under {args.frames_dir}"
    eps_ok, nf, nskip = set(), 0, 0
    t0 = time.time()
    frs = list(frames(files, args.per_ep))
    total = len(frs)
    for fr in frs:
        try:
            caught.clear(); stash.clear(); stash["arm"] = True
            stash["ep"], stash["fi"] = fr["ep"], fr.get("fi", -1)
            infer_with_grad(fr)
            if not stash.get("done"):
                raise RuntimeError("post_dit was not differentiated; every G would be zero")
            eps_ok.add(fr["ep"]); nf += 1
        except torch.cuda.OutOfMemoryError:
            nskip += 1
            torch.cuda.empty_cache()
        if nf % 10 == 0 and nf:
            print(f"[aog] {nf}/{total} frames ({nskip} skipped) {(time.time() - t0) / 60:.1f}m "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)

    cov = len(eps_ok) / max(1, len(files))
    print(f"\n[aog] {nf}/{total} frames succeeded ({nskip} skipped); "
          f"{len(eps_ok)}/{len(files)} episodes represented ({100 * cov:.0f}%)", flush=True)
    if cov < 0.9:
        raise SystemExit(f"only {len(eps_ok)}/{len(files)} episodes contributed")
    n_touched = sum(1 for n in TGT if trG[n] > 0)
    print(f"[aog] layers with nonzero trG: {n_touched}/{len(TGT)}", flush=True)
    if n_touched < 0.9 * len(TGT):
        raise SystemExit(f"gradients reached only {n_touched}/{len(TGT)} action Linears")

    # Nystrom -> V per layer, in the smoothed+rotated basis.
    mot.to("cpu"); torch.cuda.empty_cache()
    AABS = torch.load(args.absmax, map_location="cpu", weights_only=False)
    absmax = AABS["absmax"] if "absmax" in AABS else AABS
    subs, skipped = {}, []
    for n, m in TGT.items():
        S = sketch[n]
        if float(S.abs().sum()) == 0.0 or n not in absmax:
            skipped.append(n); continue
        W = m.weight.data.detach().cpu().float()
        s = compute_smooth(absmax[n].float().cpu(), W, ALPHA)
        B = hadamard_block(m.in_features, FWHT_BLOCK_MAX)
        Ssm = s.view(-1, 1) * S
        Sr = block_fwht(Ssm.t().contiguous(), B).t().contiguous() if B >= 4 else Ssm
        om = omega(n, m.in_features, torch.device("cpu"))
        omr = block_fwht((s.view(-1, 1) * om).t().contiguous(), B).t().contiguous() if B >= 4 else om
        C = omr.t() @ Sr
        C = 0.5 * (C + C.t())
        try:
            ev, U = torch.linalg.eigh(C.double())
            keep = U[:, -args.rank:].flip(-1).float()
            V, _ = torch.linalg.qr(Sr @ keep)
            subs[n] = {"V": V[:, :args.rank].contiguous().half(),
                       "evals": ev.flip(0)[:args.rank].float()}
        except Exception as e:
            skipped.append(n); print(f"[aog] {n}: eig failed ({e})", flush=True)

    meta = {"rank": args.rank, "sketch_q": Q, "frames": nf, "frames_requested": total,
            "frames_oom_skipped": nskip, "per_ep": args.per_ep,
            "episode_files": len(files), "episodes_represented": len(eps_ok),
            "coverage": round(cov, 4), "skipped_layers": len(skipped), "seed": args.seed,
            "alpha": ALPHA, "expert": "action", "model": "imagewam-flux2-klein-4b",
            "basis": "smoothed+rotated", "protocol": "c50",
            "metric": "d(action chunk)/d(x_l) at one denoising step of the action expert; the "
                      "video KV cache is prefilled ONCE and is constant w.r.t. this gradient"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"subspaces": subs, "trG": trG, "meta": meta}, args.out)
    print(f"[aog] wrote {args.out}: {len(subs)} layers with V, {len(skipped)} skipped "
          f"({skipped}), {nf}/{total} frames over {len(eps_ok)}/{len(files)} episodes", flush=True)
    Path(args.out).with_name(Path(args.out).stem + "_meta.json").write_text(
        json.dumps({"meta": meta, "n_with_V": len(subs), "skipped": skipped[:20]}, indent=1))


if __name__ == "__main__":
    main()
