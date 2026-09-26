"""Per-input-channel activation absmax of ImageWAM's MoT Linears (smoothing calibration).

Protocol: one bf16 closed-loop clean RoboTwin episode per task (50 tasks). The RoboTwin manager runs
one worker process per task; each worker records the elementwise max of |x| over every call of every
target Linear and writes one shard at exit. merge_shards takes the elementwise max over shards,
which is exact for an absmax.

  IW_CALIB_OUT=<dir>/absmax  ->  hooks installed by the patched policy, shard written at exit
"""
from __future__ import annotations

import atexit
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

# All MoT Linears; the frozen text encoder and the VAE are not part of the MoT.
TARGET_RE = re.compile(r"^mixtures\.(video|action)\.")
# Linears that execute during RoboTwin inference (the 2 video final_layer Linears never run).
EXPECTED_LAYERS = 154


def target_linears(mot) -> dict:
    return {n: m for n, m in mot.named_modules()
            if isinstance(m, nn.Linear) and TARGET_RE.match(n)}


def install_absmax_hooks(model, out_prefix: str, tag: str = "") -> int:
    """Record per-input-channel absmax on every target Linear; write a shard at process exit."""
    mot = getattr(model, "mot", model)
    tgt = target_linears(mot)
    if not tgt:
        raise RuntimeError("no mixtures.* Linears found; unexpected model layout")

    stats: dict[str, torch.Tensor] = {}
    seen = {"n": 0}

    def mk(name):
        def hook(mod, inp):
            x = inp[0]
            if not torch.is_tensor(x):
                return None
            a = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0).float().cpu()
            prev = stats.get(name)
            stats[name] = a if prev is None else torch.maximum(prev, a)
            seen["n"] += 1
            return None
        return hook

    for n, m in tgt.items():
        m.register_forward_pre_hook(mk(n))

    def dump():
        if not stats:
            print("[iw-calib] WARNING: no activations recorded; writing nothing", flush=True)
            return
        p = Path(f"{out_prefix}_{tag or 'x'}_{os.getpid()}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"absmax": stats, "n_layers": len(stats), "n_calls": seen["n"],
                    "tag": tag, "pid": os.getpid()}, p)
        print(f"[iw-calib] wrote {p} ({len(stats)} layers, {seen['n']} forward calls)", flush=True)

    atexit.register(dump)
    print(f"[iw-calib] absmax hooks on {len(tgt)} Linears (tag={tag!r})", flush=True)
    return len(tgt)


def merge_shards(shard_dir: str, prefix: str, out: str, expect_layers: int = EXPECTED_LAYERS):
    """Elementwise max over the shards `<shard_dir>/<prefix>_*.pt`; checks task and layer coverage."""
    d = Path(shard_dir)
    shards = sorted(d.glob(f"{Path(prefix).name}_*.pt"))
    if not shards:
        raise SystemExit(f"no shards matching {prefix}_*.pt in {d}")
    merged: dict[str, torch.Tensor] = {}
    tags, calls = set(), 0
    for s in shards:
        b = torch.load(s, map_location="cpu", weights_only=False)
        tags.add(b.get("tag", "?"))
        calls += int(b.get("n_calls", 0))
        for k, v in b["absmax"].items():
            prev = merged.get(k)
            merged[k] = v if prev is None else torch.maximum(prev, v)
    if len(tags) < 0.9 * len(shards):
        raise SystemExit(f"shards cover only {len(tags)} distinct tasks over {len(shards)} files")
    if len(merged) != expect_layers:
        raise SystemExit(f"merged {len(merged)} layers, expected {expect_layers}")
    torch.save({"absmax": merged, "n_layers": len(merged), "n_shards": len(shards),
                "tasks": sorted(tags), "n_calls": calls,
                "protocol": "1 closed-loop bf16 episode per RoboTwin task, elementwise max over shards"},
               out)
    print(f"[iw-calib] merged {len(shards)} shards / {len(tags)} tasks -> {out} "
          f"({len(merged)} layers, {calls} forward calls)", flush=True)
    return out
