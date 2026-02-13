"""
Fine-tune a pretrained ViT teacher on CIFAR-100.

This creates a CIFAR-100-adapted teacher checkpoint for NSA-ViT distillation.

Usage:
    python -m nsa_vit.train_teacher
    python -m nsa_vit.train_teacher --model vit_small_patch16_224 --epochs 20
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsa_vit.models.vit_teacher import get_device
from nsa_vit.utils.metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_data_loaders(image_size: int, batch_size: int,
                     data_root: str, num_workers: int):
    """Create CIFAR-100 data loaders with ViT-appropriate transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True,
        transform=train_transform)
    val_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True,
        transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train_teacher(model_name: str = 'vit_small_patch16_224',
                  num_classes: int = 100,
                  epochs: int = 20,
                  batch_size: int = 64,
                  lr: float = 1e-4,
                  weight_decay: float = 0.05,
                  data_root: str = './data',
                  checkpoint_dir: str = './checkpoints',
                  device_str: str = 'auto',
                  num_workers: int = 4,
                  seed: int = 42):
    """Fine-tune a pretrained ViT on CIFAR-100."""

    # Setup
    torch.manual_seed(seed)
    device = get_device(device_str)
    logger.info(f"Device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load pretrained model and adapt classifier
    model = timm.create_model(model_name, pretrained=True,
                              num_classes=num_classes)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Teacher: {model_name}, {param_count/1e6:.2f}M parameters")

    # Data
    image_size = model.default_cfg.get('input_size', (3, 224, 224))[-1]
    train_loader, val_loader = get_data_loaders(
        image_size, batch_size, data_root, num_workers)
    logger.info(f"CIFAR-100: {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val")

    # Optimizer: lower LR for pretrained backbone, higher for head
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr * 10},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start

        # Validate
        val_accs = evaluate(model, val_loader, device, top_k=(1, 5))

        log_msg = (f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) "
                   f"loss={avg_loss:.4f} train_acc={train_acc:.2f}% "
                   f"val_top1={val_accs[1]:.2f}% val_top5={val_accs[5]:.2f}%")

        if val_accs[1] > best_acc:
            best_acc = val_accs[1]
            save_path = os.path.join(checkpoint_dir, 'teacher_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'model_name': model_name,
                'num_classes': num_classes,
            }, save_path)
            log_msg += " *best*"

        logger.info(log_msg)

    logger.info(f"Teacher fine-tuning complete. Best val top-1: {best_acc:.2f}%")
    logger.info(f"Checkpoint saved to: {os.path.join(checkpoint_dir, 'teacher_best.pth')}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune ViT teacher on CIFAR-100')
    parser.add_argument('--model', type=str, default='vit_small_patch16_224')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_teacher(
        model_name=args.model,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
