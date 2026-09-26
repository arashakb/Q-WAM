"""Block-diagonal orthonormal Hadamard rotation.

A layer with input width d uses blocks of size B = the largest power of two dividing d, capped at
1024, so no padding is needed. The rotation is applied to the smoothed activation and, once and
offline, to the smoothed weight; since H is orthogonal and symmetric, x~ W~^T equals x W^T.
"""
from __future__ import annotations

import math

import torch


def fwht(x):
    """Orthonormal fast Walsh-Hadamard transform along the last dim (size must be a power of 2)."""
    orig = x.shape; m = orig[-1]; x = x.reshape(-1, m).clone(); h = 1
    while h < m:
        x = x.view(-1, m // (2 * h), 2, h); a = x[:, :, 0, :]; b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2).reshape(-1, m); h *= 2
    return (x / math.sqrt(m)).reshape(orig)


_HMAT_CACHE = {}


def hadamard_matrix(B, dtype, device):
    """Orthonormal Hadamard matrix H with x @ H == fwht(x), cached per (B, dtype, device).

    Built by applying fwht to the identity, so the matrix form and the butterfly form share the
    same ordering and 1/sqrt(B) normalization.
    """
    key = (B, dtype, str(device))
    if key not in _HMAT_CACHE:
        eye = torch.eye(B, device=device, dtype=torch.float32)
        _HMAT_CACHE[key] = fwht(eye).to(dtype).contiguous()
    return _HMAT_CACHE[key]


def block_fwht(x, B):
    """Block-diagonal orthonormal Hadamard along the last dim (D % B == 0), as one GEMM."""
    *lead, D = x.shape
    if D % B:
        return fwht(x.reshape(*lead, D // B, B)).reshape(*lead, D)
    H = hadamard_matrix(B, x.dtype, x.device)
    return (x.reshape(*lead, D // B, B) @ H).reshape(*lead, D)


def hadamard_block(in_f, cap=1024):
    """Largest power-of-2 block size that divides in_f (so the block Hadamard needs no padding), capped."""
    B = in_f & (-in_f)               # lowest set bit = largest power of 2 dividing in_f
    return min(B, cap)


# Short aliases used throughout the model integrations.
_fwht = fwht
_hadamard_matrix = hadamard_matrix
_bfwht = block_fwht
_afq_block = hadamard_block
