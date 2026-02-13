"""
Teacher ViT model loading and hook-based activation capture.

Uses timm to load pretrained ViTs. Registers forward hooks to
capture attention maps and intermediate activations needed for
NSA distillation losses, without modifying timm source code.
"""

import logging
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device with auto-detection."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_teacher(model_name: str, num_classes: int,
                   pretrained: bool = True,
                   checkpoint_path: str = None) -> nn.Module:
    """
    Load a pretrained ViT teacher from timm.

    Disables fused attention so explicit attention weights are computed
    and can be captured via hooks.

    Args:
        model_name: timm model name (e.g. "vit_small_patch16_224")
        num_classes: number of output classes
        pretrained: use ImageNet pretrained weights
        checkpoint_path: optional path to a fine-tuned checkpoint

    Returns:
        teacher model in eval mode with requires_grad=False
    """
    teacher = timm.create_model(model_name, pretrained=pretrained,
                                num_classes=num_classes)

    # Disable fused attention so we get explicit attention weight matrices
    for block in teacher.blocks:
        if hasattr(block.attn, 'fused_attn'):
            block.attn.fused_attn = False

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        teacher.load_state_dict(state_dict)
        logger.info(f"Loaded teacher checkpoint from {checkpoint_path}")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logger.info(f"Teacher: {model_name}, embed_dim={teacher.embed_dim}, "
                f"depth={len(teacher.blocks)}, "
                f"heads={teacher.blocks[0].attn.num_heads}")
    return teacher


class TeacherWrapper:
    """
    Wraps a timm ViT teacher with hooks to capture intermediate activations.

    Captured outputs:
    - attn_maps: list of (B, H, N, N) attention weight matrices per block
    - attn_value_outputs: list of (B, N, D) A @ V before out_proj per block
    - pre_ffn_inputs: list of (B, N, D) post-norm2 activations (input to FFN)
    - ffn_intermediates: list of (B, N, 4D) post-GELU activations inside FFN
    - cls_features: (B, D) final CLS token after norm, before head

    Usage:
        wrapper = TeacherWrapper(teacher)
        outputs = wrapper(images)
        # outputs["logits"], outputs["attn_maps"], etc.
    """

    def __init__(self, teacher: nn.Module):
        self.teacher = teacher
        self.hooks = []
        self._intermediates = {}
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on teacher blocks."""
        for l, block in enumerate(self.teacher.blocks):
            # Hook on norm2 to capture pre-FFN input (post-LN2 activation)
            self.hooks.append(
                block.norm2.register_forward_hook(
                    self._make_hook(f"norm2_{l}"))
            )

            # Hook on attention module to capture attention weights
            # timm Attention.forward returns x, but we need attn weights
            # We hook the softmax output by hooking the attn_drop layer
            # which receives the attention weights after softmax
            self.hooks.append(
                block.attn.attn_drop.register_forward_hook(
                    self._make_hook(f"attn_weights_{l}"))
            )

            # Hook on MLP activation (GELU) to capture post-GELU output
            # timm's Mlp: fc1 -> act -> drop1 -> fc2 -> drop2
            self.hooks.append(
                block.mlp.act.register_forward_hook(
                    self._make_hook(f"mlp_act_{l}"))
            )

            # Hook on attention proj input to capture A @ V (value output before out_proj)
            self.hooks.append(
                block.attn.proj.register_forward_hook(
                    self._make_input_hook(f"attn_value_out_{l}"))
            )

        # Hook on final LayerNorm to capture CLS features for L_global
        self.hooks.append(
            self.teacher.norm.register_forward_hook(
                self._make_hook("final_norm"))
        )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._intermediates[name] = output.detach()
        return hook

    def _make_input_hook(self, name: str):
        """Capture module input (for proj: the A @ V tensor)."""
        def hook(module, input, output):
            self._intermediates[name] = input[0].detach()
        return hook

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict:
        """
        Forward pass through teacher, capturing intermediates.

        Returns dict with keys: logits, attn_maps, attn_value_outputs,
        pre_ffn_inputs, ffn_intermediates, cls_features.
        """
        self._intermediates.clear()
        logits = self.teacher(images)

        num_blocks = len(self.teacher.blocks)
        attn_maps = []
        attn_value_outputs = []
        pre_ffn_inputs = []
        ffn_intermediates = []

        for l in range(num_blocks):
            attn_key = f"attn_weights_{l}"
            if attn_key in self._intermediates:
                attn_maps.append(self._intermediates[attn_key])

            val_key = f"attn_value_out_{l}"
            if val_key in self._intermediates:
                attn_value_outputs.append(self._intermediates[val_key])

            norm2_key = f"norm2_{l}"
            if norm2_key in self._intermediates:
                pre_ffn_inputs.append(self._intermediates[norm2_key])

            act_key = f"mlp_act_{l}"
            if act_key in self._intermediates:
                ffn_intermediates.append(self._intermediates[act_key])

        cls_features = self._intermediates.get("final_norm")
        if cls_features is not None:
            cls_features = cls_features[:, 0, :].clone()

        return {
            'logits': logits,
            'attn_maps': attn_maps,
            'attn_value_outputs': attn_value_outputs,
            'pre_ffn_inputs': pre_ffn_inputs,
            'ffn_intermediates': ffn_intermediates,
            'cls_features': cls_features,
        }

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
