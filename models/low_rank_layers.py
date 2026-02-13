"""
Low-rank linear layer: the core building block for NSA-ViT compression.

Replaces nn.Linear(in_features, out_features) with two smaller matrices
U ∈ R^{out×r} and V ∈ R^{in×r} such that W_hat = U @ V^T.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LowRankLinear(nn.Module):
    """
    Low-rank factorization of a linear layer.

    Replaces W ∈ R^{out x in} with U ∈ R^{out x r}, V ∈ R^{in x r}
    so that W_hat = U @ V^T.

    Forward: out = (x @ V) @ U^T + bias

    PyTorch nn.Linear convention: stores W^T internally, computes x @ W^T.
    Our decomposition: W = U @ V^T where U ∈ R^{out x r}, V ∈ R^{in x r}.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.U)
        nn.init.kaiming_uniform_(self.V)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, ..., in_features)
        z = x @ self.V          # (*, in) @ (in, r) -> (*, r)
        out = z @ self.U.t()    # (*, r) @ (r, out) -> (*, out)
        if self.bias is not None:
            out = out + self.bias
        return out

    @property
    def effective_weight(self) -> Tensor:
        """Reconstruct W_hat = U @ V^T for null-space loss computation."""
        return self.U @ self.V.t()  # (out, in)

    @staticmethod
    def from_pretrained(linear: nn.Linear, rank: int) -> 'LowRankLinear':
        """
        Initialize from a pretrained nn.Linear via truncated SVD.

        Absorbs singular values symmetrically: U_r * sqrt(S), V_r * sqrt(S).
        """
        W = linear.weight.data  # (out_features, in_features)
        U_full, S, Vh = torch.linalg.svd(W, full_matrices=False)

        U_r = U_full[:, :rank]       # (out, r)
        S_r = S[:rank]               # (r,)
        V_r = Vh[:rank, :].t()       # (in, r)

        sqrt_S = torch.sqrt(S_r)

        layer = LowRankLinear(
            linear.in_features, linear.out_features, rank,
            bias=(linear.bias is not None)
        )
        layer.U.data = U_r * sqrt_S.unsqueeze(0)   # (out, r)
        layer.V.data = V_r * sqrt_S.unsqueeze(0)    # (in, r)

        if linear.bias is not None:
            layer.bias.data = linear.bias.data.clone()

        return layer

    @staticmethod
    def from_weight(weight: Tensor, rank: int,
                    bias: Tensor = None) -> 'LowRankLinear':
        """
        Initialize from a raw weight tensor via truncated SVD.

        Useful for per-head initialization where we have weight slices
        rather than nn.Linear modules.
        """
        out_features, in_features = weight.shape
        U_full, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        U_r = U_full[:, :rank]
        S_r = S[:rank]
        V_r = Vh[:rank, :].t()

        sqrt_S = torch.sqrt(S_r)

        layer = LowRankLinear(in_features, out_features, rank,
                              bias=(bias is not None))
        layer.U.data = U_r * sqrt_S.unsqueeze(0)
        layer.V.data = V_r * sqrt_S.unsqueeze(0)

        if bias is not None:
            layer.bias.data = bias.clone()

        return layer

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'rank={self.rank}, bias={self.bias is not None}')
