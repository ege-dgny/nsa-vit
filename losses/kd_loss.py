"""
Knowledge distillation loss on output logits.

Supports KL divergence and 1-Wasserstein distance between
softened teacher and student output distributions.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def kd_loss(student_logits: Tensor, teacher_logits: Tensor,
            temperature: float = 4.0,
            metric: str = "wasserstein") -> Tensor:
    """
    Knowledge distillation loss.

    Args:
        student_logits: (B, num_classes)
        teacher_logits: (B, num_classes) — will be detached
        temperature: softening temperature
        metric: "kl" or "wasserstein"

    Returns:
        scalar loss
    """
    if metric == "kl":
        p = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        q = F.log_softmax(student_logits / temperature, dim=-1)
        loss = F.kl_div(q, p, reduction='batchmean') * (temperature ** 2)

    elif metric == "wasserstein":
        p = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        q = F.softmax(student_logits / temperature, dim=-1)

        # 1-Wasserstein via sorted CDF difference
        p_sorted, _ = torch.sort(p, dim=-1)
        q_sorted, _ = torch.sort(q, dim=-1)
        cdf_p = torch.cumsum(p_sorted, dim=-1)
        cdf_q = torch.cumsum(q_sorted, dim=-1)
        loss = (cdf_p - cdf_q).abs().mean()
    else:
        raise ValueError(f"Unknown KD metric: {metric}. Use 'kl' or 'wasserstein'.")

    return loss
