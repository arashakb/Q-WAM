"""Activation absmax calibration for the smoothing factors of LingBot-VA.

Every frame of the calibration episodes goes through the server's _infer with frame_st_id=0 (full
frame-state reset per frame, instruction encoded once per episode). A forward pre-hook on each block
Linear keeps the running per-input-channel maximum of |x| over all tokens, denoising steps (video
and action) and frames. Calibration is sharded by episode (files[shard::nshard]); shards are merged
by an elementwise maximum.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .export import target_linears
from .harness import PromptCache, build_obs


def install_absmax_hooks(targets: dict, acc: dict) -> list:
    """Pre-hooks writing the running per-channel max |input| of each module into acc[name]."""
    def mk(name):
        def hook(mod, inp):
            x = inp[0].detach()
            c = x.float().abs().amax(dim=tuple(range(x.ndim - 1)))
            acc[name] = c if acc[name] is None else torch.maximum(acc[name], c.to(acc[name].device))
        return hook
    return [m.register_forward_pre_hook(mk(n)) for n, m in targets.items()]


def iter_frames(files, max_frames: int = 0):
    """Every frame of every episode file, in order; new_ep marks an episode's first frame."""
    n = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        prompt = str(d["prompt"])
        H, L, R, S = d["head"], d["left"], d["right"], d["state"]
        for i in range(H.shape[0]):
            yield dict(head=H[i], left=L[i], right=R[i], state=S[i], prompt=prompt, new_ep=(i == 0))
            n += 1
            if max_frames and n >= max_frames:
                return


def partial_path(out_path) -> Path:
    out = Path(out_path)
    return out.with_name(out.stem + ".partial.pt")


def calibrate_absmax(server, files, out_path, *, empty_every: int = 1, ckpt_every: int = 250,
                     max_run_skips: int = 25, park_text_encoder: bool = True, max_frames: int = 0,
                     log=print) -> dict:
    """Run the calibration frames of `files` and write {layer: absmax [in]} to out_path.

    Intermediate snapshots go to <out>.partial.pt, so an interrupted run never leaves a file at
    out_path. A frame that runs out of GPU memory is skipped; `max_run_skips` consecutive skips abort.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = target_linears(server.transformer)
    log(f"[calib] {len(targets)} target Linears, "
        f"{sum(m.weight.numel() for m in targets.values()) / 1e9:.2f} B params; {len(files)} episodes")
    prompts = PromptCache(server, park=park_text_encoder)

    acc = {n: None for n in targets}
    hooks = install_absmax_hooks(targets, acc)

    def snapshot():
        return {n: (v.cpu() if torch.is_tensor(v) else None) for n, v in acc.items()}

    def save_partial(nframes, nskip):
        torch.save(snapshot(), partial_path(out_path))
        log(f"[calib] snapshot: absmax over {nframes} frames ({nskip} skipped) -> "
            f"{partial_path(out_path)}")

    t0, k, skipped, run_skips = time.time(), 0, 0, 0
    try:
        for ob in iter_frames(files, max_frames):
            # Recover outside the except block: a live exception keeps the failed call's CUDA
            # tensors referenced, and empty_cache() would free nothing.
            oom_msg, fatal = None, None
            try:
                prompts.reset(ob["prompt"], force=ob["new_ep"])
                with torch.no_grad():
                    server._infer(build_obs(ob), frame_st_id=0)
            except torch.cuda.OutOfMemoryError as e:
                oom_msg = str(e).splitlines()[0]
            except Exception as e:                               # noqa: BLE001 - re-raised below
                fatal = f"{type(e).__name__}: {e}"
            if fatal is not None:
                save_partial(k, skipped)
                raise RuntimeError(f"_infer failed on frame {k + skipped}: {fatal}")
            if oom_msg is not None:
                skipped += 1
                run_skips += 1
                prompts.invalidate()                             # clean _reset(prompt) next frame
                gc.collect()
                torch.cuda.empty_cache()
                if run_skips <= 3 or skipped % 100 == 0:
                    log(f"[calib] OOM on frame {k + skipped}, skipped={skipped} (run {run_skips}): "
                        f"{oom_msg}")
                if run_skips >= max_run_skips:
                    save_partial(k, skipped)
                    raise RuntimeError(f"{run_skips} consecutive OOM skips; memory is not recovering")
                continue
            run_skips = 0
            k += 1
            if ckpt_every and k % ckpt_every == 0:
                save_partial(k, skipped)
            if empty_every and k % empty_every == 0:
                torch.cuda.empty_cache()
            if k % 5 == 0:
                el = time.time() - t0
                log(f"[calib] {k} frames ({el:.0f}s, {el / k:.2f}s/frame, "
                    f"alloc {torch.cuda.memory_allocated() / 2**30:.1f}GB "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)")
    finally:
        for h in hooks:
            h.remove()

    absmax = snapshot()
    torch.save(absmax, out_path)
    stats = {"frames": k, "frames_oom_skipped": skipped, "episode_files": [Path(f).name for f in files],
             "layers_with_data": sum(1 for v in absmax.values() if torch.is_tensor(v))}
    out_path.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    partial_path(out_path).unlink(missing_ok=True)
    log(f"[calib] absmax over {k} frames ({skipped} skipped, "
        f"{100 * k / max(1, k + skipped):.1f}% coverage) -> {out_path}")
    return absmax


def merge_shards(shards: list) -> dict:
    """Elementwise max over per-shard {layer: absmax} dicts (None entries are skipped)."""
    merged, nlayers = {}, None
    for d in shards:
        if nlayers is None:
            nlayers = len(d)
        for k, v in d.items():
            if not torch.is_tensor(v):
                continue
            merged[k] = v if k not in merged else torch.maximum(merged[k], v)
    if len(merged) < (nlayers or 0):
        raise ValueError(f"only {len(merged)} of {nlayers} layers got calibration data")
    return merged
