"""
W&B Sweep for NSA-ViT hyperparameter tuning.

Sweeps over compression-related (and optionally other) hyperparameters to find
maximum compression without compromising accuracy. Each run logs to wandb with
compression metrics (params, FLOPs, ratio) and validation accuracy.

Usage:
    # Register sweep and run locally (runs multiple training runs)
    python -m nsa_vit.sweep --config nsa_vit/configs/default.yaml --sweep_project nsa-vit

    # Or with custom sweep config (fewer epochs for quick search)
    python -m nsa_vit.sweep --config nsa_vit/configs/default.yaml --sweep_project nsa-vit --epochs 5

Edit the SWEEP_CONFIG dict below to add parameters (e.g. alpha, gamma, epochs)
or change method (grid, random, bayes).
"""

import argparse
import os
import sys

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from nsa_vit.training.distill import run_distillation


# Default sweep: compression knobs to find max compression vs accuracy tradeoff
SWEEP_CONFIG = {
    "method": "grid",  # or "random", "bayes"
    "metric": {"name": "best_val_top1", "goal": "maximize"},
    "parameters": {
        "rank_selection_method": {"values": ["energy_threshold"]},
        "energy_threshold": {"values": [0.3, 0.4, 0.5, 0.6, 0.7]},
        # Alternative: sweep fixed_rank_ratio instead
        # "rank_selection_method": {"values": ["fixed_ratio"]},
        # "fixed_rank_ratio": {"values": [0.15, 0.2, 0.25, 0.3]},
    },
}
# SWEEP_CONFIG = {
#     "method": "grid",
#     "metric": {"name": "best_val_top1", "goal": "maximize"},
#     "parameters": {
#         "rank_selection_method": {"values": ["fixed_ratio"]},
#         "fixed_rank_ratio": {"values": [0.15, 0.20, 0.25, 0.30, 0.35]},
#     },
# }



def parse_args():
    parser = argparse.ArgumentParser(description="NSA-ViT W&B Sweep")
    parser.add_argument("--config", type=str, default="nsa_vit/configs/default.yaml",
                        help="Path to base YAML config")
    parser.add_argument("--sweep_project", type=str, default="nsa-vit",
                        help="W&B project for the sweep")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (team or user)")
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of runs (for random/bayes); default = all for grid")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs in config (e.g. 5 for quick sweep)")
    return parser.parse_args()


def make_run_function(config_path: str, extra_overrides: dict):
    """Return a function that runs one sweep trial with wandb.config merged into config."""
    def run_fn():
        # Explicitly initialize a run so wandb.config is available across wandb versions.
        run = wandb.init()
        try:
            overrides = dict(run.config)
            overrides["use_wandb"] = True
            overrides.update(extra_overrides)
            run_distillation(config_path, overrides=overrides)
        finally:
            wandb.finish()
    return run_fn


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    extra_overrides = {}
    if args.epochs is not None:
        extra_overrides["epochs"] = args.epochs

    sweep_config = SWEEP_CONFIG.copy()
    if args.count is not None and sweep_config["method"] != "grid":
        sweep_config["count"] = args.count

    sweep_id = wandb.sweep(sweep_config, project=args.sweep_project, entity=args.entity)
    run_fn = make_run_function(config_path, extra_overrides)
    wandb.agent(sweep_id, function=run_fn)


if __name__ == "__main__":
    main()
