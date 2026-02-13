"""
Attention map distillation loss.

Matches student and teacher attention patterns via KL (recommended) or MSE.
Replaces null-space loss for attention projections because the
softmax nonlinearity breaks the linear null-space assumption.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def attention_map_loss(student_attn_maps: list, teacher_attn_maps: list,
                       every_n: int = 1, metric: str = "kl") -> Tensor:
    """
    Compute attention map distillation loss.

    When metric='kl': L_attn = (1/L) * sum_l KL(A_T^l || A_S^l) (per-row distributions).
    When metric='mse': L_attn = (1/L) * sum_l ||A_T^l - A_S^l||_F^2

    Args:
        student_attn_maps: list of L tensors, each (B, H, N, N)
        teacher_attn_maps: list of L tensors, each (B, H, N, N)
        every_n: only compute loss every N blocks (memory optimization)
        metric: "kl" (default, math-aligned) or "mse"

    Returns:
        scalar loss
    """
    if not student_attn_maps or not teacher_attn_maps:
        return torch.tensor(0.0)

    device = student_attn_maps[0].device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for l, (a_s, a_t) in enumerate(
            zip(student_attn_maps, teacher_attn_maps)):
        if l % every_n != 0:
            continue
        if metric == "kl":
            # Each row of A is a probability distribution; KL(A_T || A_S)
            p = a_t.detach()
            log_q = F.log_softmax(a_s, dim=-1)
            loss = loss + F.kl_div(log_q, p, reduction="batchmean")
        else:
            loss = loss + ((a_t.detach() - a_s) ** 2).mean()
        count += 1

    return loss / max(count, 1)


def value_output_loss(teacher_value_outputs: list, student_value_outputs: list,
                      every_n: int = 1) -> Tensor:
    """
    Value-output matching loss: sum_l ||A_l^T V_l^T - A_l^S V_l^S||_F^2.

    Each list entry is (B, N, D): the attention @ value output before out_proj.

    Args:
        teacher_value_outputs: list of L tensors (B, N, D)
        student_value_outputs: list of L tensors (B, N, D)
        every_n: only compute loss every N blocks (memory optimization)

    Returns:
        scalar loss (mean over blocks used)
    """
    if not teacher_value_outputs or not student_value_outputs:
        return torch.tensor(0.0)

    device = student_value_outputs[0].device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for l, (t_out, s_out) in enumerate(
            zip(teacher_value_outputs, student_value_outputs)):
        if l % every_n != 0:
            continue
        loss = loss + ((t_out.detach() - s_out) ** 2).mean()
        count += 1

    return loss / max(count, 1)
