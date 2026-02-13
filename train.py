"""
NSA-ViT distillation training entry point.

Usage:
    python -m nsa_vit.train --config configs/default.yaml
    python -m nsa_vit.train --config configs/default.yaml --epochs 10 --lr 5e-5
"""

import argparse
import sys
import os

# Ensure the parent directory is on the path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsa_vit.training.distill import run_distillation


def parse_args():
    parser = argparse.ArgumentParser(
        description='NSA-ViT: Null-Space Absorbing ViT Compression')
    parser.add_argument('--config', type=str,
                        default='nsa_vit/configs/default.yaml',
                        help='Path to YAML config file')

    # Override any config value from command line
    parser.add_argument('--teacher_model', type=str, default=None)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--energy_threshold', type=float, default=None)
    parser.add_argument('--rank_selection_method', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None,
                        help='Null-space loss weight')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Attention distillation loss weight')
    parser.add_argument('--beta', type=float, default=None,
                        help='KD loss weight')
    parser.add_argument('--lambda_orth', type=float, default=None,
                        help='Orthonormality loss weight')

    return parser.parse_args()


def main():
    args = parse_args()

    # Collect non-None overrides
    overrides = {}
    for key in ['teacher_model', 'teacher_checkpoint', 'num_classes',
                'dataset', 'data_root', 'batch_size', 'epochs', 'lr',
                'device', 'energy_threshold', 'rank_selection_method',
                'seed', 'alpha', 'gamma', 'beta', 'lambda_orth']:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    run_distillation(args.config, overrides=overrides if overrides else None)


if __name__ == '__main__':
    main()
