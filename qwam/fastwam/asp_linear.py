"""Simulated W4A4 linear layer with Action-Subspace Protection (ASP).

Offline, from the bf16 weight W [d_out, d_in], the per-input-channel activation absmax a and the
layer's Action Observability Gramian G (AOG, [d_in, d_in]):

    s     = a^alpha / max|W|^(1 - alpha)       SmoothQuant factor per input channel (s = 1 if no a)
    W~    = W diag(s) H                        H: block-diagonal Hadamard along d_in
    V     = top-r eigenvectors of H diag(s) G diag(s) H                    [d_in, r]
    W~V   kept in 16 bits                                                   [d_out, r]
    Wq    = Q4(W~ - (W~V) V^T)                 the deflated weight, INT4 per group of 32 along d_in

Forward, with x~ = H diag(s)^-1 x the tensor the quantizer sees:

    c = x~ V                                   r coefficients per token, 16 bits
    y = Q4(x~ - c V^T) Wq^T + c (W~V)^T + b

A layer built without an AOG (rank 0) is plain smoothed and rotated W4A4: y = Q4(x~) Q4(W~)^T + b.

This is simulated quantization: Q4 rounds to the INT4 grid and dequantizes at once, and the GEMMs
run in the model dtype on the dequantized operands. Activations use one dynamic scale per token and
group of channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwam.hadamard import block_fwht, hadamard_block
from qwam.quant import compute_smooth, group_quant_dequant


class ASPLinear(nn.Module):
    """Drop-in replacement for an ``nn.Linear`` that simulates W4A4 inference with ASP.

    Args:
        linear: the bf16 layer to replace (its weight and bias are read, not modified).
        act_absmax: per-input-channel activation absmax [d_in]; None disables smoothing (s = 1).
        aog: dense AOG of the layer input [d_in, d_in]; None or ``rank=0`` disables ASP.
        rank: number of protected directions r.
        rotate: apply the block-diagonal Hadamard rotation.
        alpha: SmoothQuant migration strength.
        w_bits, w_group: weight bits and group size (groups run along d_in).
        a_bits, a_group: activation bits and group size (per token, along the channel dim).
        hadamard_cap: largest Hadamard block; the block is the largest power of two dividing d_in.
    """

    def __init__(self, linear: nn.Linear, act_absmax: torch.Tensor | None = None,
                 aog: torch.Tensor | None = None, rank: int = 32, *, rotate: bool = True,
                 alpha: float = 0.5, w_bits: int = 4, w_group: int = 32, a_bits: int = 4,
                 a_group: int = 32, hadamard_cap: int = 1024):
        super().__init__()
        self.compute_dtype = linear.weight.dtype
        device = linear.weight.device
        W = linear.weight.data.float()
        d_out, d_in = W.shape
        self.in_features, self.out_features = d_in, d_out
        self.a_bits, self.a_group = a_bits, a_group

        s = compute_smooth(act_absmax, W, alpha)
        Wt = W * s.unsqueeze(0)
        self.block = hadamard_block(d_in, hadamard_cap)
        self.rotate = bool(rotate) and self.block >= 4
        if self.rotate:
            Wt = block_fwht(Wt, self.block)

        V = None
        if rank > 0 and aog is not None:
            G = aog.float().to(device)
            if G.shape != (d_in, d_in):
                raise ValueError(f"AOG has shape {tuple(G.shape)}; expected a dense ({d_in}, {d_in}) matrix")
            # Express the AOG in the coordinates of x~: H diag(s) G diag(s) H.
            G = s.view(-1, 1) * G * s.view(1, -1)
            if self.rotate:
                G = block_fwht(block_fwht(G.t(), self.block).t(), self.block)
            _, eigvecs = torch.linalg.eigh(G)                 # ascending eigenvalues
            V = eigvecs.flip(-1)[:, :rank].contiguous()
            del G
        self.rank = 0 if V is None else V.shape[1]

        if V is None:
            Wq = group_quant_dequant(Wt, w_bits, w_group, dim=1)
            self.register_buffer("asp_basis", None)
            self.register_buffer("asp_weight", None)
        else:
            WV = Wt @ V                                       # the protected part of the weight
            Wq = group_quant_dequant(Wt - WV @ V.t(), w_bits, w_group, dim=1)
            self.register_buffer("asp_basis", V.to(self.compute_dtype))
            self.register_buffer("asp_weight", WV.to(self.compute_dtype))
        self.register_buffer("inv_smooth", (1.0 / s).to(self.compute_dtype))
        self.register_buffer("weight_q", Wq.to(self.compute_dtype))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        xt = (x * self.inv_smooth.to(x.dtype)).to(dt)
        if self.rotate:
            xt = block_fwht(xt.float(), self.block).to(dt)
        if self.asp_basis is None:
            xq = group_quant_dequant(xt.float(), self.a_bits, self.a_group, dim=-1).to(dt)
            y = xq @ self.weight_q.t()
        else:
            c = xt @ self.asp_basis                           # protected coefficients, 16 bits
            x_perp = xt - c @ self.asp_basis.t()              # remainder, orthogonal to span(V)
            xq = group_quant_dequant(x_perp.float(), self.a_bits, self.a_group, dim=-1).to(dt)
            y = xq @ self.weight_q.t() + c @ self.asp_weight.t()
        return y if self.bias is None else y + self.bias.to(dt)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"asp_rank={self.rank}, rotate={self.rotate} (block {self.block}), "
                f"a_bits={self.a_bits}, a_group={self.a_group}")
