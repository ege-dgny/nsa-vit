"""
NSA-ViT training loop with all loss components.

Handles teacher forward (frozen, with hooks), student forward,
loss computation (CE + NSA + attention + value-output + KD + orthonormality + global CLS),
and optimization with gradient clipping. Supports Mixup/CutMix augmentation.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..losses.null_space_loss import compute_block_null_space_loss
from ..losses.attention_distill import attention_map_loss, value_output_loss
from ..losses.orthonormality_loss import orthonormality_loss
from ..losses.kd_loss import kd_loss
from ..losses.global_loss import global_cls_loss
from ..utils.metrics import evaluate

logger = logging.getLogger(__name__)


class NSAViTTrainer:
    """
    Trainer for NSA-ViT distillation.

    Combines seven loss components:
        L = L_ce + alpha*L_null + gamma*L_attn + eta*L_val + beta*L_kd
            + lambda_orth*L_orth + mu*L_global
    """

    def __init__(self, teacher_wrapper, student: nn.Module,
                 train_loader, val_loader, config: dict, device):
        self.teacher = teacher_wrapper
        self.student = student.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss weights
        self.alpha = config.get('alpha', 0.5)
        self.gamma = config.get('gamma', 0.1)
        self.eta = config.get('eta', 0.1)
        self.beta = config.get('beta', 1.0)
        self.lambda_orth = config.get('lambda_orth', 0.01)
        self.mu = config.get('mu', 0.01)
        self.kd_temperature = config.get('kd_temperature', 4.0)
        self.kd_metric = config.get('kd_metric', 'kl')
        self.attn_loss_every_n = config.get('attn_loss_every_n', 3)
        self.attn_loss_metric = config.get('attn_loss_metric', 'kl')
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.label_smoothing = config.get('label_smoothing', 0.1)

        # Mixup / CutMix
        mixup_alpha = config.get('mixup_alpha', 0.0)
        cutmix_alpha = config.get('cutmix_alpha', 0.0)
        self.mixup_fn = None
        self.soft_ce_fn = None
        if mixup_alpha > 0 or cutmix_alpha > 0:
            try:
                from timm.data import Mixup
                from timm.loss import SoftTargetCrossEntropy
                num_classes = config.get('num_classes', 100)
                self.mixup_fn = Mixup(
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                    label_smoothing=self.label_smoothing,
                    num_classes=num_classes,
                )
                self.soft_ce_fn = SoftTargetCrossEntropy()
                logger.info(f"Mixup enabled: mixup_alpha={mixup_alpha}, "
                            f"cutmix_alpha={cutmix_alpha}")
            except ImportError:
                logger.warning("timm.data.Mixup not available; "
                               "disabling Mixup/CutMix.")

        # Optimizer
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 0.05)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler with warmup
        epochs = config.get('epochs', 100)
        warmup_epochs = config.get('warmup_epochs', 3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs)
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs)

        # Logging
        log_dir = config.get('log_dir', './runs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every = config.get('save_every', 10)
        self.eval_every = config.get('eval_every', 1)

        self.use_wandb = config.get('use_wandb', False)
        self._wandb_available = False
        if self.use_wandb:
            try:
                import wandb
                self._wandb_available = wandb.run is not None
            except ImportError:
                pass

        self.best_val_acc = 0.0
        self.best_val_top5 = 0.0

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns loss dict."""
        self.student.train()
        total_losses = {
            'total': 0.0, 'ce': 0.0, 'null': 0.0,
            'attn': 0.0, 'val': 0.0, 'kd': 0.0, 'orth': 0.0, 'global': 0.0,
        }
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher forward on clean images (frozen, with hooks)
            teacher_out = self.teacher(images)

            # Apply Mixup/CutMix to images and labels for student
            mixed_labels = None
            if self.mixup_fn is not None:
                images, mixed_labels = self.mixup_fn(images, labels)

            # Student forward (with intermediates)
            student_out = self.student(images, return_intermediates=True)

            # 1. Supervised CE loss
            if mixed_labels is not None:
                L_ce = self.soft_ce_fn(student_out['logits'], mixed_labels)
            else:
                L_ce = F.cross_entropy(student_out['logits'], labels,
                                       label_smoothing=self.label_smoothing)

            # 2. Null-space loss (FFN junctions)
            L_null = compute_block_null_space_loss(
                student_out, teacher_out, self.student)

            # 3. Attention map distillation (KL or MSE)
            L_attn = attention_map_loss(
                student_out['attn_maps'], teacher_out['attn_maps'],
                every_n=self.attn_loss_every_n, metric=self.attn_loss_metric)

            # 4. Value-output matching (A @ V per block)
            L_val = value_output_loss(
                teacher_out.get('attn_value_outputs', []),
                student_out.get('attn_value_outputs', []),
                every_n=self.attn_loss_every_n)

            # 5. Output KD
            L_kd = kd_loss(
                student_out['logits'], teacher_out['logits'],
                temperature=self.kd_temperature, metric=self.kd_metric)

            # 6. Orthonormality regularization
            L_orth = orthonormality_loss(self.student)

            # 7. Global CLS matching
            L_global = global_cls_loss(
                teacher_out.get('cls_features'),
                student_out.get('cls_features'))

            # Combined loss
            loss = (L_ce
                    + self.alpha * L_null
                    + self.gamma * L_attn
                    + self.eta * L_val
                    + self.beta * L_kd
                    + self.lambda_orth * L_orth
                    + self.mu * L_global)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.gradient_clip)
            self.optimizer.step()

            # Accumulate losses
            total_losses['total'] += loss.item()
            total_losses['ce'] += L_ce.item()
            total_losses['null'] += L_null.item()
            total_losses['attn'] += L_attn.item()
            total_losses['val'] += L_val.item()
            total_losses['kd'] += L_kd.item()
            total_losses['orth'] += L_orth.item()
            total_losses['global'] += L_global.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"E{epoch} B{batch_idx+1}/{len(self.train_loader)} "
                    f"Tot:{loss.item():.4f} CE:{L_ce.item():.4f} "
                    f"Null:{L_null.item():.4f} Attn:{L_attn.item():.4f} "
                    f"Val:{L_val.item():.4f} KD:{L_kd.item():.4f} "
                    f"Orth:{L_orth.item():.4f} Global:{L_global.item():.4f}")

        # Average losses
        avg_losses = {k: v / max(num_batches, 1)
                      for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Evaluate on validation set."""
        accs = evaluate(self.student, self.val_loader, self.device,
                        top_k=(1, 5))
        return {'val_top1': accs[1], 'val_top5': accs[5]}

    def train(self, num_epochs: int):
        """Full training loop."""
        logger.info(f"Starting NSA-ViT training for {num_epochs} epochs")
        logger.info(f"Loss weights: alpha={self.alpha}, gamma={self.gamma}, "
                    f"eta={self.eta}, beta={self.beta}, lambda_orth={self.lambda_orth}, "
                    f"mu={self.mu}, attn_metric={self.attn_loss_metric}")

        for epoch in range(1, num_epochs + 1):
            start = time.time()

            # Train
            train_losses = self.train_one_epoch(epoch)
            elapsed = time.time() - start

            # LR scheduling
            if epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log training losses
            for k, v in train_losses.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            self.writer.add_scalar('train/lr', current_lr, epoch)
            if self.use_wandb and self._wandb_available:
                import wandb
                wandb_log = {f'train/{k}': v for k, v in train_losses.items()}
                wandb_log['train/lr'] = current_lr
                wandb.log(wandb_log, step=epoch)

            log_msg = (f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) "
                       f"lr={current_lr:.6f} ")
            log_msg += " ".join(f"{k}={v:.4f}" for k, v in train_losses.items())

            # Validate
            if epoch % self.eval_every == 0:
                val_metrics = self.validate()
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                if self.use_wandb and self._wandb_available:
                    import wandb
                    wandb.log({'val/val_top1': val_metrics['val_top1'],
                               'val/val_top5': val_metrics['val_top5']}, step=epoch)

                log_msg += (f" | val_top1={val_metrics['val_top1']:.2f}% "
                            f"val_top5={val_metrics['val_top5']:.2f}%")

                # Save best
                if val_metrics['val_top1'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_top1']
                    self._save_checkpoint(epoch, is_best=True)
                    log_msg += " *best*"
                self.best_val_top5 = max(self.best_val_top5, val_metrics['val_top5'])

            logger.info(log_msg)

            # Periodic save
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        # Final save
        self._save_checkpoint(num_epochs, is_best=False)
        logger.info(f"Training complete. Best val top-1: "
                     f"{self.best_val_acc:.2f}%")
        self.writer.close()
        if self.use_wandb and self._wandb_available:
            import wandb
            wandb.run.summary['best_val_top1'] = self.best_val_acc
            wandb.run.summary['best_val_top5'] = self.best_val_top5

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pth')
        else:
            path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
