"""
Orthonormality regularization loss.

Encourages the U factor of each LowRankLinear to remain orthonormal,
preventing rank collapse and maintaining well-conditioned factorizations.

L_orth = (1/N) * sum_i ||U_i^T U_i - I||_F^2
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..models.low_rank_layers import LowRankLinear


def orthonormality_loss(model: nn.Module) -> Tensor:
    """
    Compute orthonormality regularization over all LowRankLinear.U matrices.

    Returns:
        scalar loss averaged over all LowRankLinear modules
    """
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for module in model.modules():
        if isinstance(module, LowRankLinear):
            U = module.U  # (out, r)
            UtU = U.t() @ U  # (r, r)
            I = torch.eye(U.shape[1], device=U.device)
            loss = loss + ((UtU - I) ** 2).mean()
            count += 1

    return loss / max(count, 1)
