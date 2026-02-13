"""
Low-rank Vision Transformer student model for NSA-ViT compression.

All nn.Linear layers in attention and MLP are replaced with LowRankLinear.
Patch embedding, classification head, and LayerNorms remain full-rank.
The forward pass returns intermediate activations needed for NSA losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .low_rank_layers import LowRankLinear


class LowRankAttention(nn.Module):
    """
    Multi-head attention with low-rank Q, K, V, and output projections.

    Uses separate LowRankLinear for Q, K, V (not fused) to allow
    per-head SVD and different ranks per projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Placeholder LowRankLinear — initialized via svd_init
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> tuple:
        """
        Args:
            x: (B, N, D) input tokens

        Returns:
            output: (B, N, D)
            attn_weights: (B, H, N, N) attention weight matrix
            value_output: (B, N, D) attention @ value before out_proj (for L_val)
        """
        B, N, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, d_k).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, d_k).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, d_k).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_weights = attn.softmax(dim=-1)
        attn_weights_dropped = self.attn_drop(attn_weights)

        # Value output (A @ V) before out_proj — use attn_weights (no dropout) for L_val
        value_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)

        out = (attn_weights_dropped @ v)  # (B, H, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out, attn_weights, value_output


class LowRankMLP(nn.Module):
    """
    Feed-forward network with low-rank fc1 and fc2 layers.

    Structure: fc1 -> GELU -> dropout -> fc2 -> dropout
    """

    def __init__(self, embed_dim: int, mlp_ratio: int = 4,
                 drop: float = 0.0):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio

        # Placeholder — initialized via svd_init
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor,
                return_intermediate: bool = False) -> tuple:
        """
        Args:
            x: (B, N, D)
            return_intermediate: if True, also return post-GELU activation

        Returns:
            output: (B, N, D)
            intermediate: (B, N, 4D) post-GELU activation, or None
        """
        h = self.fc1(x)
        h = self.act(h)
        intermediate = h if return_intermediate else None
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return h, intermediate


class LowRankViTBlock(nn.Module):
    """
    Transformer block with low-rank attention and MLP.

    Structure:
        x' = x + Attention(LN1(x))
        x'' = x' + MLP(LN2(x'))
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: int = 4, drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LowRankAttention(embed_dim, num_heads,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = LowRankMLP(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: Tensor,
                return_intermediates: bool = False) -> tuple:
        """
        Args:
            x: (B, N, D)
            return_intermediates: capture activations for loss computation

        Returns:
            x: (B, N, D) block output
            intermediates: dict with keys:
                - attn_weights: (B, H, N, N)
                - attn_value_output: (B, N, D) A @ V before out_proj
                - pre_ffn_input: (B, N, D) post-norm2, pre-MLP
                - ffn_intermediate: (B, N, 4D) post-GELU inside MLP
              or empty dict if return_intermediates=False
        """
        intermediates = {}

        # Attention branch
        attn_out, attn_weights, attn_value_output = self.attn(self.norm1(x))
        x = x + attn_out

        if return_intermediates:
            intermediates['attn_weights'] = attn_weights
            intermediates['attn_value_output'] = attn_value_output

        # FFN branch
        norm2_out = self.norm2(x)

        if return_intermediates:
            intermediates['pre_ffn_input'] = norm2_out

        mlp_out, ffn_mid = self.mlp(norm2_out,
                                    return_intermediate=return_intermediates)
        x = x + mlp_out

        if return_intermediates and ffn_mid is not None:
            intermediates['ffn_intermediate'] = ffn_mid

        return x, intermediates


class LowRankViT(nn.Module):
    """
    Low-rank Vision Transformer for NSA-ViT compression.

    Full-rank components (copied from teacher):
    - patch_embed: Conv2d patch embedding
    - cls_token, pos_embed: token and positional embeddings
    - norm: final LayerNorm
    - head: classification head

    Low-rank components (initialized via SVD):
    - blocks[*].attn.{q_proj, k_proj, v_proj, out_proj}: LowRankLinear
    - blocks[*].mlp.{fc1, fc2}: LowRankLinear
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 100,
                 embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, mlp_ratio: int = 4,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        num_patches = (img_size // patch_size) ** 2

        # Full-rank components — mirrors timm's PatchEmbed structure
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Low-rank transformer blocks
        self.blocks = nn.ModuleList([
            LowRankViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                            drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: Tensor,
                         return_intermediates: bool = False) -> tuple:
        """Extract features through patch embedding and transformer blocks."""
        B = x.shape[0]

        # Patch embedding (Sequential wrapping Conv2d)
        x = self.patch_embed[0](x)      # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N, D) where N = N_patches + 1

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        all_intermediates = []
        for block in self.blocks:
            x, intermediates = block(x, return_intermediates)
            if return_intermediates:
                all_intermediates.append(intermediates)

        x = self.norm(x)
        return x, all_intermediates

    def forward(self, x: Tensor,
                return_intermediates: bool = False):
        """
        Forward pass.

        When return_intermediates=False:
            Returns logits tensor (B, num_classes)

        When return_intermediates=True:
            Returns dict with:
                logits: (B, num_classes)
                attn_maps: list of (B, H, N, N) per block
                attn_value_outputs: list of (B, N, D) per block
                pre_ffn_inputs: list of (B, N, D) per block
                ffn_intermediates: list of (B, N, 4D) per block
                cls_features: (B, D) final CLS token after norm, before head
        """
        x, all_intermediates = self.forward_features(
            x, return_intermediates)

        logits = self.head(x[:, 0])  # CLS token

        if not return_intermediates:
            return logits

        attn_maps = [d['attn_weights'] for d in all_intermediates]
        attn_value_outputs = [d['attn_value_output'] for d in all_intermediates]
        pre_ffn_inputs = [d['pre_ffn_input'] for d in all_intermediates]
        ffn_intermediates = [d['ffn_intermediate'] for d in all_intermediates]

        return {
            'logits': logits,
            'attn_maps': attn_maps,
            'attn_value_outputs': attn_value_outputs,
            'pre_ffn_inputs': pre_ffn_inputs,
            'ffn_intermediates': ffn_intermediates,
            'cls_features': x[:, 0, :],
        }


def create_student_from_teacher(teacher: nn.Module,
                                config: dict) -> LowRankViT:
    """
    Create a LowRankViT student initialized from a teacher model.

    Args:
        teacher: pretrained timm ViT model
        config: configuration dict

    Returns:
        LowRankViT with compressed layers initialized via SVD
    """
    from ..utils.svd_init import initialize_student_from_teacher

    embed_dim = teacher.embed_dim
    num_heads = teacher.blocks[0].attn.num_heads
    depth = len(teacher.blocks)
    mlp_ratio = teacher.blocks[0].mlp.fc1.out_features // embed_dim

    student = LowRankViT(
        img_size=config.get('image_size', 224),
        patch_size=config.get('patch_size', 16),
        num_classes=config.get('num_classes', 100),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    initialize_student_from_teacher(teacher, student, config)
    return student
