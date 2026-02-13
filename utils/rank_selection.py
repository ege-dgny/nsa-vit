"""
Rank selection utilities for low-rank compression.

Determines truncation rank for each weight matrix based on
spectral energy thresholds or fixed ratios.
"""

import torch
from torch import Tensor


def compute_layer_rank(weight: Tensor, method: str = "energy_threshold",
                       **kwargs) -> int:
    """
    Determine truncation rank for a weight matrix.

    Args:
        weight: (out_features, in_features) tensor
        method: "energy_threshold" or "fixed_ratio"
        kwargs:
            threshold (float): for energy_threshold, default 0.95
            ratio (float): for fixed_ratio, default 0.25

    Returns:
        rank (int), clamped to [1, min(m, n)]
    """
    max_rank = min(weight.shape)
    S = torch.linalg.svdvals(weight)
    total_energy = (S ** 2).sum()

    if method == "energy_threshold":
        threshold = kwargs.get("threshold", 0.95)
        cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
        indices = (cumulative >= threshold).nonzero(as_tuple=True)[0]
        rank = int(indices[0].item()) + 1 if len(indices) > 0 else max_rank
    elif method == "fixed_ratio":
        ratio = kwargs.get("ratio", 0.25)
        rank = max(1, int(max_rank * ratio))
    else:
        raise ValueError(f"Unknown rank selection method: {method}")

    return min(rank, max_rank)


def analyze_ranks(model, method: str = "energy_threshold", **kwargs) -> dict:
    """
    Analyze the effective rank of all linear layers in a model.

    Returns dict mapping layer name -> {rank, max_rank, ratio, energy_retained}.
    """
    results = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.dim() == 2:
            W = module.weight.data
            max_rank = min(W.shape)
            rank = compute_layer_rank(W, method, **kwargs)

            S = torch.linalg.svdvals(W)
            total_energy = (S ** 2).sum()
            retained = (S[:rank] ** 2).sum() / total_energy

            results[name] = {
                'rank': rank,
                'max_rank': max_rank,
                'ratio': rank / max_rank,
                'energy_retained': retained.item(),
                'shape': tuple(W.shape),
            }
    return results
