"""
Global CLS matching loss for NSA-ViT.

Bounds cumulative residual error by matching final CLS token
representations (after all blocks and final LayerNorm, before head).
"""

import torch
from torch import Tensor


def global_cls_loss(teacher_cls: Tensor, student_cls: Tensor) -> Tensor:
    """
    L_global = ||z_L^T - z_L^S||_2^2 (mean over batch and dim).

    Args:
        teacher_cls: (B, D) final CLS from teacher
        student_cls: (B, D) final CLS from student

    Returns:
        scalar loss
    """
    if teacher_cls is None or student_cls is None:
        dev = student_cls.device if student_cls is not None else torch.device("cpu")
        return torch.tensor(0.0, device=dev)
    return ((teacher_cls.detach() - student_cls) ** 2).mean()
