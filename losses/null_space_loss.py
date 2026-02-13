"""
Null-space absorption loss for NSA-ViT.

Drives the approximation error (h_T - h_S) into the null space of the
downstream weight matrix, so it does not affect later layer outputs.

L_null = ||W_next @ (h_T - h_S)||^2

Applied at FFN junctions only (not attention, due to softmax nonlinearity):
  Point A: pre-FFN (post-norm2) -> W_next = fc1
  Point B: mid-FFN (post-GELU) -> W_next = fc2
"""

import torch
from torch import Tensor

from ..models.low_rank_layers import LowRankLinear


def null_space_loss(student_act: Tensor, teacher_act: Tensor,
                    next_layer_weight: Tensor) -> Tensor:
    """
    Compute null-space loss using the full effective weight matrix.

    Args:
        student_act: (B, N, D) student activation
        teacher_act: (B, N, D) teacher activation (detached)
        next_layer_weight: (out, in) effective weight of next layer

    Returns:
        scalar loss
    """
    e = teacher_act.detach() - student_act  # (B, N, D)
    We = torch.einsum('oi,bni->bno', next_layer_weight, e)  # (B, N, out)
    return (We ** 2).mean()


def null_space_loss_efficient(student_act: Tensor, teacher_act: Tensor,
                              U: Tensor, V: Tensor) -> Tensor:
    """
    Compute null-space loss WITHOUT materializing W_hat = U @ V^T.

    Uses two smaller matmuls through the rank-r bottleneck:
        Ve = V^T @ e    -> (B, N, r)
        UVe = U @ Ve    -> (B, N, out)

    Cost: O(r * D * N) instead of O(D^2 + D * N).

    Args:
        student_act: (B, N, D) student activation
        teacher_act: (B, N, D) teacher activation (detached)
        U: (out, r) left factor of next layer
        V: (in=D, r) right factor of next layer

    Returns:
        scalar loss
    """
    e = teacher_act.detach() - student_act  # (B, N, D)
    Ve = torch.einsum('dr,bnd->bnr', V, e)    # (B, N, r)
    UVe = torch.einsum('or,bnr->bno', U, Ve)  # (B, N, out)
    return (UVe ** 2).mean()


def compute_block_null_space_loss(student_out: dict, teacher_out: dict,
                                  student_model) -> Tensor:
    """
    Compute total null-space loss across all transformer blocks.

    Two NSA loss points per block:
    - Point A: post-norm2 (pre-FFN) activation, W_next = fc1
    - Point B: post-GELU (mid-FFN) activation, W_next = fc2

    Uses efficient factored computation when layers are LowRankLinear.
    """
    num_blocks = len(student_out['pre_ffn_inputs'])
    total_loss = torch.tensor(0.0, device=student_out['logits'].device)
    count = 0

    for l in range(num_blocks):
        s_block = student_model.blocks[l]

        # Point A: pre-FFN -> fc1
        s_pre_ffn = student_out['pre_ffn_inputs'][l]
        t_pre_ffn = teacher_out['pre_ffn_inputs'][l]
        fc1 = s_block.mlp.fc1

        if isinstance(fc1, LowRankLinear):
            total_loss = total_loss + null_space_loss_efficient(
                s_pre_ffn, t_pre_ffn, fc1.U, fc1.V)
        else:
            total_loss = total_loss + null_space_loss(
                s_pre_ffn, t_pre_ffn, fc1.weight)
        count += 1

        # Point B: mid-FFN (post-GELU) -> fc2
        s_ffn_mid = student_out['ffn_intermediates'][l]
        t_ffn_mid = teacher_out['ffn_intermediates'][l]
        fc2 = s_block.mlp.fc2

        if isinstance(fc2, LowRankLinear):
            total_loss = total_loss + null_space_loss_efficient(
                s_ffn_mid, t_ffn_mid, fc2.U, fc2.V)
        else:
            total_loss = total_loss + null_space_loss(
                s_ffn_mid, t_ffn_mid, fc2.weight)
        count += 1

    return total_loss / max(count, 1)
