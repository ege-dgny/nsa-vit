"""
NSA-ViT evaluation script.

Loads a trained student checkpoint and evaluates accuracy, parameter count,
FLOPs, and compression ratio.

Usage:
    python -m nsa_vit.evaluate --checkpoint checkpoints/best.pth --dataset cifar100
"""

import argparse
import logging
import os
import sys

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsa_vit.models.vit_teacher import create_teacher, get_device
from nsa_vit.models.vit_student import LowRankViT
from nsa_vit.utils.metrics import (
    evaluate, count_parameters, compute_compression_ratio,
    count_flops, print_model_summary
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate NSA-ViT student')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to student checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    device = get_device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']

    # Recreate student architecture
    teacher = create_teacher(
        model_name=config['teacher_model'],
        num_classes=config.get('num_classes', 100),
        pretrained=True,
        checkpoint_path=config.get('teacher_checkpoint'),
    )

    from nsa_vit.models.vit_student import create_student_from_teacher
    student = create_student_from_teacher(teacher, config)
    student.load_state_dict(ckpt['model_state_dict'])
    student = student.to(device).eval()

    # Print compression summary
    print_model_summary(teacher, student, device=str(device))

    # Data loader
    image_size = config.get('image_size', 224)
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    if args.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(
            root=args.data_root, train=False, download=True,
            transform=val_transform)
    elif args.dataset == 'imagenet':
        val_dir = os.path.join(args.data_root, 'val')
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Evaluate
    accs = evaluate(student, val_loader, device, top_k=(1, 5))
    print(f"\nResults:")
    print(f"  Top-1 accuracy: {accs[1]:.2f}%")
    print(f"  Top-5 accuracy: {accs[5]:.2f}%")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Best val acc during training: "
          f"{ckpt.get('best_val_acc', 'unknown')}")


if __name__ == '__main__':
    main()
