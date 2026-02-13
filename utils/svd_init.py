"""
SVD initialization utilities for creating low-rank student models
from pretrained teacher weights.

Handles per-head SVD for attention projections and standard SVD
for MLP layers.
"""

import logging
import torch
import torch.nn as nn
from torch import Tensor

from ..models.low_rank_layers import LowRankLinear
from .rank_selection import compute_layer_rank

logger = logging.getLogger(__name__)


def compute_rank_for_weight(weight: Tensor, config: dict) -> int:
    """Compute rank for a weight tensor using config settings."""
    return compute_layer_rank(
        weight,
        method=config.get('rank_selection_method', 'energy_threshold'),
        threshold=config.get('energy_threshold', 0.95),
        ratio=config.get('fixed_rank_ratio', 0.25),
    )


def init_low_rank_from_linear(linear: nn.Linear, config: dict) -> LowRankLinear:
    """Initialize a LowRankLinear from a pretrained nn.Linear."""
    rank = compute_rank_for_weight(linear.weight.data, config)
    return LowRankLinear.from_pretrained(linear, rank)


def split_qkv_weight(qkv_weight: Tensor, qkv_bias: Tensor,
                     embed_dim: int) -> tuple:
    """
    Split a fused QKV weight (3*D, D) into separate Q, K, V weights.

    Returns:
        (q_weight, k_weight, v_weight): each (D, D)
        (q_bias, k_bias, v_bias): each (D,) or None
    """
    q_w = qkv_weight[:embed_dim, :]
    k_w = qkv_weight[embed_dim:2*embed_dim, :]
    v_w = qkv_weight[2*embed_dim:3*embed_dim, :]

    if qkv_bias is not None:
        q_b = qkv_bias[:embed_dim]
        k_b = qkv_bias[embed_dim:2*embed_dim]
        v_b = qkv_bias[2*embed_dim:3*embed_dim]
    else:
        q_b = k_b = v_b = None

    return (q_w, k_w, v_w), (q_b, k_b, v_b)


def init_per_head_projection(weight: Tensor, bias: Tensor,
                             embed_dim: int, num_heads: int,
                             config: dict) -> LowRankLinear:
    """
    Initialize a LowRankLinear for one of Q/K/V using per-head SVD.

    Splits the (D, D) weight into num_heads blocks of (d_k, D),
    SVDs each independently, then reassembles into a single LowRankLinear.

    When per_head_svd is False, does a single SVD on the full weight.
    """
    if not config.get('per_head_svd', True):
        rank = compute_rank_for_weight(weight, config)
        return LowRankLinear.from_weight(weight, rank, bias)

    d_k = embed_dim // num_heads
    per_head_U = []
    per_head_V = []
    total_rank = 0

    for h in range(num_heads):
        head_weight = weight[h * d_k:(h + 1) * d_k, :]  # (d_k, D)
        rank = compute_rank_for_weight(head_weight, config)
        total_rank += rank

        U_full, S, Vh = torch.linalg.svd(head_weight, full_matrices=False)
        U_r = U_full[:, :rank]       # (d_k, r_h)
        S_r = S[:rank]               # (r_h,)
        V_r = Vh[:rank, :].t()       # (D, r_h)

        sqrt_S = torch.sqrt(S_r)
        per_head_U.append(U_r * sqrt_S.unsqueeze(0))
        per_head_V.append(V_r * sqrt_S.unsqueeze(0))

    # Assemble block-diagonal U and shared V
    # U is block-diagonal: (D, total_rank) with blocks along diagonal
    U_assembled = torch.zeros(embed_dim, total_rank)
    V_assembled = torch.zeros(embed_dim, total_rank)

    col_offset = 0
    for h in range(num_heads):
        r_h = per_head_U[h].shape[1]
        row_start = h * d_k
        row_end = (h + 1) * d_k
        U_assembled[row_start:row_end, col_offset:col_offset + r_h] = per_head_U[h]
        V_assembled[:, col_offset:col_offset + r_h] = per_head_V[h]
        col_offset += r_h

    layer = LowRankLinear(embed_dim, embed_dim, total_rank,
                          bias=(bias is not None))
    layer.U.data = U_assembled
    layer.V.data = V_assembled

    if bias is not None:
        layer.bias.data = bias.clone()

    logger.info(f"  Per-head SVD: {num_heads} heads, total rank={total_rank} "
                f"(avg {total_rank/num_heads:.0f}/head)")
    return layer


def initialize_student_from_teacher(teacher, student, config: dict):
    """
    Initialize all LowRankLinear layers in the student from teacher weights.

    Handles:
    - Attention QKV (per-head SVD) -> separate Q, K, V LowRankLinear
    - Attention output projection -> single LowRankLinear
    - MLP fc1, fc2 -> single LowRankLinear each
    - Non-compressed layers (patch_embed, head, norms) are copied directly
    """
    num_heads = teacher.blocks[0].attn.num_heads
    embed_dim = teacher.embed_dim

    # Copy non-compressed components
    # timm's patch_embed is PatchEmbed with .proj Conv2d inside;
    # student's patch_embed is Sequential(Conv2d)
    teacher_pe = teacher.patch_embed
    student.patch_embed[0].weight.data = teacher_pe.proj.weight.data.clone()
    if teacher_pe.proj.bias is not None:
        student.patch_embed[0].bias.data = teacher_pe.proj.bias.data.clone()

    student.cls_token.data = teacher.cls_token.data.clone()
    if config.get('trainable_pos_embed', True):
        student.pos_embed.data = teacher.pos_embed.data.clone()
    student.norm.load_state_dict(teacher.norm.state_dict())
    student.head.load_state_dict(teacher.head.state_dict())

    # Initialize each block's compressed layers
    for l, (t_block, s_block) in enumerate(
            zip(teacher.blocks, student.blocks)):
        logger.info(f"Initializing block {l}...")

        # Copy LayerNorms
        s_block.norm1.load_state_dict(t_block.norm1.state_dict())
        s_block.norm2.load_state_dict(t_block.norm2.state_dict())

        # --- Attention QKV (per-head SVD) ---
        qkv_weight = t_block.attn.qkv.weight.data  # (3*D, D)
        qkv_bias = t_block.attn.qkv.bias.data if t_block.attn.qkv.bias is not None else None
        (q_w, k_w, v_w), (q_b, k_b, v_b) = split_qkv_weight(
            qkv_weight, qkv_bias, embed_dim)

        logger.info(f"  Q projection:")
        s_block.attn.q_proj = init_per_head_projection(
            q_w, q_b, embed_dim, num_heads, config)
        logger.info(f"  K projection:")
        s_block.attn.k_proj = init_per_head_projection(
            k_w, k_b, embed_dim, num_heads, config)
        logger.info(f"  V projection:")
        s_block.attn.v_proj = init_per_head_projection(
            v_w, v_b, embed_dim, num_heads, config)

        # --- Attention output projection ---
        proj_linear = t_block.attn.proj
        proj_rank = compute_rank_for_weight(proj_linear.weight.data, config)
        s_block.attn.out_proj = LowRankLinear.from_pretrained(
            proj_linear, proj_rank)
        logger.info(f"  Output proj: rank={proj_rank}")

        # --- MLP fc1 and fc2 ---
        fc1_linear = t_block.mlp.fc1
        fc1_rank = compute_rank_for_weight(fc1_linear.weight.data, config)
        s_block.mlp.fc1 = LowRankLinear.from_pretrained(fc1_linear, fc1_rank)
        logger.info(f"  MLP fc1: rank={fc1_rank}")

        fc2_linear = t_block.mlp.fc2
        fc2_rank = compute_rank_for_weight(fc2_linear.weight.data, config)
        s_block.mlp.fc2 = LowRankLinear.from_pretrained(fc2_linear, fc2_rank)
        logger.info(f"  MLP fc2: rank={fc2_rank}")
