"""ASP subspaces of LingBot-VA from a sketched Action Observability Gramian (AOG).

For a block Linear l with input x_l, the AOG is G_l = E[J_l^T J_l], where J_l is the Jacobian of
the action prediction with respect to x_l. The server's sampler runs under torch.no_grad, so the
estimator differentiates the action-mode transformer calls one at a time:

  * Frames: `per_ep` evenly spaced frames of every calibration episode (4 x 50 = 200 in the paper),
    each run through _infer(frame_st_id=0) after a frame-state reset, as in calibration.
  * The video denoising loop runs without gradients. Every action-mode transformer call of the
    frame (the action denoising steps and the final cache-update call) runs with autograd enabled
    and with all transformer parameters requiring grad, so the input x_l of every block Linear is
    a node of that call's graph (the call's inputs are built without gradients and the cached
    keys/values of earlier calls are constants). Right after the call, `nprobe` Gaussian probes u
    with the shape of the call's output are backpropagated, g_l = d<out, u>/d x_l, and the output
    is returned detached, so the sampler continues as without the estimator.
  * J_l is therefore the Jacobian of one call's output, the action prediction of one denoising step
    for both classifier-free-guidance halves of the batch, with respect to x_l; it is not taken
    through the unrolled sampler. The contributions of all action-mode calls of all frames are
    summed. Layers that read the same tensor, such as the q, k and v projections of one attention,
    are handed the gradient of that shared tensor, i.e. the sum over its consumers.
  * Instead of the d x d matrix, each layer accumulates the sketch S_l = sum g g^T Omega_l with a
    Gaussian test matrix Omega_l [d, rank + oversample] seeded by the layer name, and
    tr(G_l) = sum |g|^2 in the raw basis.

After the pass (on the CPU), with s the smoothing factor of the export (alpha 0.5, calibrated absmax)
and H the export's block Hadamard: S~ = H diag(s) S, Omega~ = H diag(s) Omega,
C = sym(Omega~^T S~), U = top-rank eigenvectors of C, and V = orth(S~ U) is stored in fp16 together
with the top eigenvalues of C. V lives in the smoothed+rotated basis the INT4 branch works in.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn as nn

from qwam.hadamard import block_fwht, hadamard_block
from qwam.quant import compute_smooth

from .export import EXPECTED_LAYERS, target_linears, tensor_digest
from .harness import PromptCache, build_obs

_OMEGA = {}


def omega(name: str, d_in: int, q: int, seed: int, device) -> torch.Tensor:
    """Gaussian test matrix [d_in, q] of one layer, reproducible across processes (sha256 of the name)."""
    key = (name, d_in, q, seed)
    if key not in _OMEGA:
        h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device="cpu").manual_seed((h ^ (seed * 7919)) % (2 ** 31))
        _OMEGA[key] = torch.randn(d_in, q, generator=g)
    return _OMEGA[key].to(device)


def sample_frames(files, per_ep: int):
    """`per_ep` evenly spaced frames (np.linspace) of every episode file, in order."""
    for ep, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        p = str(d["prompt"])
        H, L, R, S = d["head"], d["left"], d["right"], d["state"]
        n = H.shape[0]
        idx = np.linspace(0, n - 1, num=min(per_ep, n), dtype=int)
        for j, i in enumerate(idx):
            yield dict(head=H[i], left=L[i], right=R[i], state=S[i], prompt=p,
                       new_ep=(j == 0), ep=ep)


def accumulate_sketches(server, files, *, rank: int = 32, oversample: int = 32, per_ep: int = 4,
                        nprobe: int = 1, seed: int = 42, park_text_encoder: bool = True, log=print):
    """Run the sampled frames and return (sketch, trG, stats); see the module docstring."""
    q = rank + oversample
    tr = server.transformer
    targets = target_linears(tr)
    log(f"[aog] {len(targets)} target Linears")
    if len(targets) != EXPECTED_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_LAYERS} target Linears, found {len(targets)}")
    for n, m in targets.items():
        if type(m) is not nn.Linear:
            raise RuntimeError(f"{n} is {type(m).__name__}: the model is already quantized")
    prompts = PromptCache(server, park=park_text_encoder)

    # Parameters must require grad: many layer inputs derive from the KV cache and the
    # conditioning rather than from the marked call inputs.
    tr.requires_grad_(True)

    caught, state = {}, {"arm": False, "calls": 0, "ncaught": 0}
    sketch = {n: torch.zeros(m.in_features, q, dtype=torch.float32) for n, m in targets.items()}
    trG = {n: 0.0 for n in targets}

    def mk(name):
        def hook(mod, inp):
            x = inp[0]
            # Inputs outside the graph (e.g. read from the detached KV cache) contribute nothing.
            if state["arm"] and torch.is_grad_enabled() and torch.is_tensor(x) and x.requires_grad:
                x.retain_grad()
                caught[name] = x
        return hook

    hooks = [m.register_forward_pre_hook(mk(n)) for n, m in targets.items()]
    orig_forward = tr.forward

    def _mark(x):
        return x.detach().requires_grad_(True) if torch.is_tensor(x) and x.is_floating_point() else x

    def forward(*a, **kw):
        if not (kw.get("action_mode") and state["arm"]):
            return orig_forward(*a, **kw)
        with torch.enable_grad():
            # A tensor, list or tuple first argument is detached and marked as a graph input.
            # LingBot-VA passes a dict, which reaches the call unchanged: its graph starts at the
            # parameters, which is why they are set to require grad above.
            first = a[0]
            if isinstance(first, list):
                first = [_mark(x) for x in first]
            elif isinstance(first, tuple):
                first = tuple(_mark(x) for x in first)
            else:
                first = _mark(first)
            out = orig_forward(*((first,) + a[1:]), **kw)
            # Differentiate now: later calls grow the rolling KV cache the graph was built on.
            if torch.is_tensor(out) and out.requires_grad and caught:
                names = list(caught)
                for p in range(nprobe):
                    u = torch.randn_like(out)
                    gs = torch.autograd.grad((out * u).sum(), [caught[n] for n in names],
                                             retain_graph=(p < nprobe - 1), allow_unused=True)
                    for nm, g in zip(names, gs):
                        if g is None:
                            continue
                        gm = g.detach().reshape(-1, g.shape[-1]).float()
                        sketch[nm] += (gm.t() @ (gm @ omega(nm, gm.shape[-1], q, seed,
                                                            gm.device))).cpu()
                        trG[nm] += float(gm.pow(2).sum())
                state["calls"] += 1
            state["ncaught"] = len(caught)
        for t in caught.values():
            t.grad = None
        caught.clear()
        return out.detach() if torch.is_tensor(out) else out

    tr.forward = forward
    total = per_ep * len(files)
    nf = nskip = 0
    eps_ok = set()
    t0 = time.time()
    try:
        for ob in sample_frames(files, per_ep):
            try:
                prompts.reset(ob["prompt"])
                caught.clear()
                state.update(arm=True, calls=0)
                server._infer(build_obs(ob), frame_st_id=0)
                state["arm"] = False
                if not state["calls"]:
                    raise RuntimeError("no action-mode call was differentiated; every G would be zero")
                if nf == 0:
                    log(f"[aog] captured {state['ncaught']}/{len(targets)} layers per action call")
                nf += 1
                eps_ok.add(ob["ep"])
                caught.clear()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                state["arm"] = False
                nskip += 1
                torch.cuda.empty_cache()
                log(f"[aog] OOM, skipped ({nskip})")
                continue
            if nf % 8 == 0:
                log(f"[aog] {nf}/{total} frames ({nskip} skipped) {(time.time() - t0) / 60:.1f}m "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    finally:
        for h in hooks:
            h.remove()
        tr.forward = orig_forward
    stats = {"frames": nf, "frames_requested": total, "frames_oom_skipped": nskip,
             "episode_files": len(files), "episodes_represented": len(eps_ok)}
    return sketch, trG, stats


def subspaces_from_sketches(sketch: dict, targets: dict, act_absmax: dict, *, rank: int = 32,
                            oversample: int = 32, seed: int = 42, alpha: float = 0.5,
                            fwht_block_max: int = 1024, log=print):
    """{layer: {"V": [in, rank] fp16, "evals": [rank]}} in the smoothed+rotated basis, and the
    list of layers without a sketch."""
    q = rank + oversample
    subs, skipped = {}, []
    for n, m in targets.items():
        S = sketch[n]
        if float(S.abs().sum()) == 0.0:
            skipped.append(n)
            continue
        W = m.weight.data.detach().cpu().float()
        aa = act_absmax.get(n)
        s = compute_smooth(aa.float().cpu() if torch.is_tensor(aa) else None, W, alpha)
        B = hadamard_block(m.in_features, fwht_block_max)
        # The transforms are linear in G, so they are applied to the sketch and the test matrix.
        Ssm = s.view(-1, 1) * S
        Sr = block_fwht(Ssm.t().contiguous(), B).t().contiguous() if B >= 4 else Ssm
        om = omega(n, m.in_features, q, seed, torch.device("cpu"))
        omr = block_fwht((s.view(-1, 1) * om).t().contiguous(), B).t().contiguous() if B >= 4 else om
        C = omr.t() @ Sr
        C = 0.5 * (C + C.t())
        try:
            ev, U = torch.linalg.eigh(C.double())
            keep = U[:, -rank:].flip(-1).float()
            V, _ = torch.linalg.qr(Sr @ keep)
            subs[n] = {"V": V[:, :rank].contiguous().half(), "evals": ev.flip(0)[:rank].float()}
        except Exception as e:                                   # noqa: BLE001 - reported below
            skipped.append(n)
            log(f"[aog] {n}: eigendecomposition failed ({e})")
    return subs, skipped


def build_subspace_file(sketch, trG, stats, targets, act_absmax, *, rank=32, oversample=32,
                        per_ep=4, nprobe=1, seed=42, alpha=0.5, fwht_block_max=1024,
                        min_coverage=0.9, log=print) -> dict:
    """The saved object {"subspaces", "trG", "meta"}; refuses runs that cover < 90% of episodes."""
    cov = stats["episodes_represented"] / max(1, stats["episode_files"])
    log(f"[aog] {stats['frames']}/{stats['frames_requested']} frames "
        f"({stats['frames_oom_skipped']} skipped); {stats['episodes_represented']}/"
        f"{stats['episode_files']} episodes represented ({100 * cov:.0f}%)")
    if cov < min_coverage:
        raise RuntimeError(f"only {stats['episodes_represented']}/{stats['episode_files']} episodes "
                           f"contributed; the subspaces would not represent the calibration tasks")
    subs, skipped = subspaces_from_sketches(sketch, targets, act_absmax, rank=rank,
                                            oversample=oversample, seed=seed, alpha=alpha,
                                            fwht_block_max=fwht_block_max, log=log)
    meta = {"rank": rank, "sketch_q": rank + oversample, **stats, "per_ep": per_ep,
            "nprobe": nprobe, "coverage": round(cov, 4), "skipped_layers": len(skipped),
            "seed": seed,
            "metric": ("per-call action Jacobian: G = sum over sampled frames and action-mode "
                       "transformer calls of J^T J, J = d(call output)/d(layer input), "
                       f"{nprobe} Gaussian probe(s) per call, Nystrom sketch"),
            "model": "lingbot-va", "basis": "smoothed+rotated", "alpha": alpha,
            "fwht_block_max": fwht_block_max, "absmax_digest": tensor_digest(act_absmax)}
    return {"subspaces": subs, "trG": trG, "meta": meta}
