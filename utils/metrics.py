"""
Evaluation metrics and model statistics for NSA-ViT.
"""

import torch
import torch.nn as nn
from torch import Tensor


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_compression_ratio(teacher: nn.Module,
                              student: nn.Module) -> float:
    """Return teacher_params / student_params."""
    t_params = count_parameters(teacher, trainable_only=False)
    s_params = count_parameters(student, trainable_only=False)
    return t_params / s_params


def count_flops(model: nn.Module, input_size=(1, 3, 224, 224),
                device='cpu') -> int:
    """
    Count FLOPs using fvcore if available, otherwise estimate.

    For LowRankLinear, FLOPs = batch * tokens * (in*r + r*out)
    vs original: batch * tokens * in * out
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(*input_size).to(device)
        model_copy = model.to(device).eval()
        flops = FlopCountAnalysis(model_copy, dummy)
        return flops.total()
    except ImportError:
        return -1  # fvcore not installed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str,
             top_k: tuple = (1, 5)) -> dict:
    """
    Evaluate model accuracy on a data loader.

    Returns dict mapping k -> accuracy percentage for each k in top_k.
    """
    model.eval()
    correct = {k: 0 for k in top_k}
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        # Handle both dict output and tensor output
        if isinstance(output, dict):
            logits = output['logits']
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        max_k = max(top_k)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (max_k, batch)
        target_expanded = labels.view(1, -1).expand_as(pred)
        matches = pred.eq(target_expanded)

        for k in top_k:
            correct[k] += matches[:k].reshape(-1).float().sum().item()
        total += labels.size(0)

    return {k: (correct[k] / total) * 100.0 for k in top_k}


def get_compression_metrics(teacher: nn.Module, student: nn.Module,
                            device: str = 'cpu') -> dict:
    """
    Return compression statistics as a dict for logging (e.g. to wandb).

    Keys: teacher_params, student_params, compression_ratio, param_savings_pct.
    If FLOPs are available: teacher_flops_G, student_flops_G, flop_reduction_pct.
    FLOP keys are omitted if fvcore is missing or FLOP count fails.
    """
    t_params = count_parameters(teacher, trainable_only=False)
    s_params = count_parameters(student, trainable_only=False)
    ratio = t_params / s_params
    param_savings_pct = (1 - 1 / ratio) * 100

    metrics = {
        'teacher_params': t_params,
        'student_params': s_params,
        'compression_ratio': ratio,
        'param_savings_pct': param_savings_pct,
    }

    t_flops = count_flops(teacher, device=device)
    s_flops = count_flops(student, device=device)
    if t_flops > 0 and s_flops > 0:
        metrics['teacher_flops_G'] = t_flops / 1e9
        metrics['student_flops_G'] = s_flops / 1e9
        metrics['flop_reduction_pct'] = (1 - s_flops / t_flops) * 100

    return metrics


def print_model_summary(teacher: nn.Module, student: nn.Module,
                        device: str = 'cpu'):
    """Print compression statistics."""
    m = get_compression_metrics(teacher, student, device=device)

    print(f"\n{'='*50}")
    print(f"Model Compression Summary")
    print(f"{'='*50}")
    print(f"Teacher parameters: {m['teacher_params']:,} ({m['teacher_params']/1e6:.2f}M)")
    print(f"Student parameters: {m['student_params']:,} ({m['student_params']/1e6:.2f}M)")
    print(f"Compression ratio:  {m['compression_ratio']:.2f}x")
    print(f"Parameter savings:  {m['param_savings_pct']:.1f}%")

    if 'teacher_flops_G' in m:
        print(f"Teacher FLOPs:      {m['teacher_flops_G']:.2f}G")
        print(f"Student FLOPs:      {m['student_flops_G']:.2f}G")
        print(f"FLOP reduction:     {m['flop_reduction_pct']:.1f}%")
    print(f"{'='*50}\n")
