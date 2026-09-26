"""Install Q-WAM on a loaded Fast-WAM model.

Every block linear of both experts (attention q/k/v/o, cross-attention q/k/v/o and the two FFN
projections; 600 layers) is replaced by an ASPLinear. Layers of the action expert, which carries
most of the action mass, get ASP with rank `rank`; video-expert layers are smoothed, rotated W4A4
without ASP. Embeddings, heads and norms stay in bf16.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .asp_linear import ASPLinear

ACTION_EXPERT = "action_expert"
VIDEO_EXPERT = "video_expert"


@dataclass(frozen=True)
class QWAMConfig:
    """Q-WAM settings; the defaults are the configuration evaluated in the paper.

    `rank`, `rotate` and `smooth` are the switches of the component ablation: rank 0 removes ASP,
    rotate=False removes the Hadamard rotation, smooth=False sets the smoothing factor to 1.
    """
    rank: int = 32          # ASP rank on the action expert (0 = no ASP)
    rotate: bool = True     # block-diagonal Hadamard rotation
    smooth: bool = True     # SmoothQuant smoothing with strength alpha
    alpha: float = 0.5
    w_bits: int = 4
    w_group: int = 32
    a_bits: int = 4
    a_group: int = 32
    hadamard_cap: int = 1024


def target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """The block linears of both experts, keyed by module name (600 on Fast-WAM)."""
    return {name: mod for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and ".blocks." in name
            and (name.startswith(VIDEO_EXPERT) or name.startswith(ACTION_EXPERT))}


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child, new)


def _load(path) -> object:
    # mmap keeps large caches out of host memory until a layer's tensors are read.
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
        return torch.load(path, map_location="cpu", weights_only=True)


def load_absmax(path) -> dict[str, torch.Tensor]:
    """Per-input-channel activation absmax {layer: [d_in]} written by merge_absmax.py."""
    d = _load(path)
    return d["absmax"] if isinstance(d, dict) and "absmax" in d else d


def load_aog(path) -> dict[str, torch.Tensor]:
    """Dense per-layer action-expert AOGs {layer: [d_in, d_in]} written by merge_aog.py."""
    g = _load(path)
    if isinstance(g, dict) and "aog" in g:
        g = g["aog"]
    for name, t in g.items():
        if not (torch.is_tensor(t) and t.ndim == 2 and t.shape[0] == t.shape[1]):
            shape = tuple(t.shape) if torch.is_tensor(t) else type(t).__name__
            raise ValueError(f"{path}: '{name}' is {shape}, not a dense square AOG")
    return g


def install_qwam(model: nn.Module, absmax: dict | None = None, aog: dict | None = None,
                 config: QWAMConfig = QWAMConfig(), verbose: bool = True) -> dict[str, int]:
    """Replace the block linears of `model` in place; returns the number of layers per kind.

    `absmax` is required when config.smooth is set, `aog` when config.rank > 0.
    """
    targets = target_linears(model)
    if not targets:
        raise RuntimeError("no block linears under video_expert/action_expert; is this a Fast-WAM model?")
    protected = [n for n in targets if config.rank > 0 and n.startswith(ACTION_EXPERT)]
    if config.smooth:
        if absmax is None:
            raise ValueError("smoothing is on but no activation absmax was given")
        missing = [n for n in targets if absmax.get(n) is None]
        if missing:
            raise KeyError(f"activation absmax missing for {len(missing)} layers, e.g. {missing[:3]}")
    if protected:
        if aog is None:
            raise ValueError(f"rank {config.rank} needs the action-expert AOG")
        missing = [n for n in protected if n not in aog]
        if missing:
            raise KeyError(f"AOG missing for {len(missing)} action-expert layers, e.g. {missing[:3]}")

    kwargs = dict(rotate=config.rotate, alpha=config.alpha, w_bits=config.w_bits,
                  w_group=config.w_group, a_bits=config.a_bits, a_group=config.a_group,
                  hadamard_cap=config.hadamard_cap)
    protected = set(protected)
    n_layers = len(targets)
    for name in list(targets):
        linear = targets.pop(name)      # drop the reference so the bf16 weight is freed on replace
        device = linear.weight.device
        a = absmax[name].to(device) if config.smooth else None
        if name in protected:
            new = ASPLinear(linear, a, aog[name].to(device), config.rank, **kwargs)
        else:
            new = ASPLinear(linear, a, None, 0, **kwargs)
        _set_submodule(model, name, new)
        del linear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    counts = {"asp": len(protected), "no_asp": n_layers - len(protected)}
    if verbose:
        print(f"[qwam] installed W{config.w_bits}A{config.a_bits} (group {config.w_group}/{config.a_group}, "
              f"smoothing={'on' if config.smooth else 'off'}, rotation={'on' if config.rotate else 'off'}): "
              f"{counts['asp']} action-expert layers with ASP rank {config.rank} + "
              f"{counts['no_asp']} layers without ASP", flush=True)
    return counts


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={os.environ[name]!r} is not a boolean")


def _env_path(name: str) -> str:
    p = os.environ.get(name, "").strip()
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(f"{name} must point to an existing file (got {p!r})")
    return p


def install_from_env(model: nn.Module, verbose: bool = True) -> dict[str, int]:
    """Entry point of the RoboTwin policy hook (QWAM_ENABLE=1).

    Reads QWAM_ABSMAX and QWAM_AOG (cache paths) and the ablation switches QWAM_RANK, QWAM_ROTATE
    and QWAM_SMOOTH; unset switches keep the paper configuration.
    """
    base = QWAMConfig()
    rank = os.environ.get("QWAM_RANK", "").strip()
    config = QWAMConfig(rank=int(rank) if rank else base.rank,
                        rotate=_env_bool("QWAM_ROTATE", base.rotate),
                        smooth=_env_bool("QWAM_SMOOTH", base.smooth))
    absmax = aog = None
    if config.smooth:
        path = _env_path("QWAM_ABSMAX")
        absmax = load_absmax(path)
        if verbose:
            print(f"[qwam] activation absmax <- {path} ({len(absmax)} layers)", flush=True)
    if config.rank > 0:
        path = _env_path("QWAM_AOG")
        aog = load_aog(path)
        if verbose:
            print(f"[qwam] action AOG <- {path} ({len(aog)} layers)", flush=True)
    if verbose:
        print(f"[qwam] config {asdict(config)}", flush=True)
    return install_qwam(model, absmax, aog, config, verbose=verbose)
