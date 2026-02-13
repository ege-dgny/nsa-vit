# NSA-ViT: Null-Space Absorbing Vision Transformer Compression

Compress pretrained Vision Transformers via low-rank factorization with null-space absorption loss. Extension of the NSA-Net method ([Low-Rank Compression of Neural Network Weights by Null-Space Encouragement](../Low-Rank_Compression_of_Neural_Network_Weights_by_Null-Space_Encouragement.pdf)) from CNNs to ViTs.

## Method

1. **SVD Initialization**: Replace each `nn.Linear` in the ViT with a `LowRankLinear(U, V)` initialized via truncated SVD of the teacher's weights. Per-head SVD is used for attention projections.

2. **Null-Space Loss**: During training, the error between teacher and student activations is projected through the next layer's weight matrix. Minimizing `||W_next @ (h_T - h_S)||²` drives errors into the null space of downstream layers, where they don't affect outputs. Applied at FFN junctions only.

3. **Attention Distillation**: Since softmax breaks the linear null-space assumption, attention maps are matched directly via MSE: `||A_T - A_S||_F²`.

4. **Combined Training**: Total loss = CE + null-space + attention distillation + output KD + orthonormality regularization.

## Setup

```bash
conda create -n nsa_vit python=3.10 -y
conda activate nsa_vit
pip install -r requirements.txt
```

## Usage

### 1. Fine-tune Teacher on CIFAR-100

```bash
python -m nsa_vit.train_teacher --model vit_small_patch16_224 --epochs 20
```

This saves a checkpoint to `checkpoints/teacher_best.pth`.

### 2. Run NSA-ViT Distillation

```bash
# Using the fine-tuned teacher
python -m nsa_vit.train --config nsa_vit/configs/default.yaml \
    --teacher_checkpoint checkpoints/teacher_best.pth

# Using timm's pretrained weights directly
python -m nsa_vit.train --config nsa_vit/configs/default.yaml

# Override hyperparameters from CLI
python -m nsa_vit.train --config nsa_vit/configs/default.yaml \
    --epochs 10 --lr 5e-5 --alpha 0.2 --rank_selection_method fixed_ratio
```

### 3. Evaluate

```bash
python -m nsa_vit.evaluate --checkpoint checkpoints/best.pth --dataset cifar100
```

## Configuration

All hyperparameters are in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher_model` | `vit_small_patch16_224` | timm model name |
| `rank_selection_method` | `energy_threshold` | `energy_threshold` or `fixed_ratio` |
| `energy_threshold` | `0.95` | Retain this fraction of spectral energy |
| `fixed_rank_ratio` | `0.25` | Fraction of min(m,n) to retain |
| `per_head_svd` | `true` | SVD attention weights per head |
| `alpha` | `0.1` | Null-space loss weight |
| `gamma` | `0.05` | Attention distillation weight |
| `beta` | `0.1` | Output KD weight |
| `lambda_orth` | `0.1` | Orthonormality regularization weight |
| `kd_temperature` | `4.0` | KD softening temperature |
| `kd_metric` | `wasserstein` | `kl` or `wasserstein` |
| `device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |

## Architecture

```
Teacher (timm ViT-Small, frozen)
        │
        ├── Forward hooks capture:
        │   ├── Attention weights (B, H, N, N)
        │   ├── Post-LayerNorm activations (B, N, D)
        │   └── Post-GELU activations (B, N, 4D)
        │
Student (LowRankViT)
        │
        ├── patch_embed: full-rank Conv2d (copied from teacher)
        ├── cls_token, pos_embed: copied, trainable
        ├── blocks × 12:
        │   ├── norm1 → LowRankAttention (separate Q, K, V LowRankLinear)
        │   ├── norm2 → LowRankMLP (fc1, fc2 as LowRankLinear)
        │   └── Residual connections preserved
        ├── norm: full-rank LayerNorm
        └── head: full-rank classifier
```

## Weights & Biases

Training and compression metrics can be logged to [Weights & Biases](https://wandb.ai) for experiment tracking and hyperparameter tuning.

**Enable logging:** In `configs/default.yaml` set `use_wandb: true`, and optionally `wandb_project` and `wandb_entity`. Each run will log train/val metrics (losses, accuracy) and **compression metrics** in the run summary: teacher/student parameter counts, compression ratio, parameter savings %, and when fvcore is available, FLOPs and FLOP reduction %. Use the wandb dashboard to compare runs and filter by compression ratio or accuracy.

**Hyperparameter sweeps:** To search for the maximum compression without compromising accuracy, run a sweep over compression-related knobs (e.g. `energy_threshold` or `fixed_rank_ratio`):

```bash
python -m nsa_vit.sweep --config nsa_vit/configs/default.yaml --sweep_project nsa-vit
```

Edit `nsa_vit/sweep.py` to change the sweep parameters (e.g. add `alpha`, `gamma`, `epochs`) or use fewer `--epochs` for a quick search. In the wandb project, use the runs table and charts to compare compression ratio vs validation accuracy and pick the best tradeoff.

## Compression Results (ViT-Small on CIFAR-100)

| Rank Method | Student Params | Compression | Notes |
|-------------|---------------|-------------|-------|
| `energy_threshold=0.95` | 25.8M | 0.84x | Retains too much rank for ViT-Small |
| `fixed_ratio=0.50` | 16.4M | 1.32x | Moderate compression |
| `fixed_ratio=0.25` | 8.4M | 2.57x | Recommended starting point |
| `energy_threshold=0.80` | 17.8M | 1.22x | Lower energy threshold |

## References

- Original NSA paper: `../Low-Rank_Compression_of_Neural_Network_Weights_by_Null-Space_Encouragement.pdf`
- Mathematical analysis: `../nsa_vit_comprehensive_evaluation.md`
- Implementation spec: `../NSA_ViT_Implementation_Spec.md`
- CNN reference: `../low_rank_decom/Kaggle_CIFAR100_Alexnet/student_alexnet.py`
