"""
Full NSA-ViT distillation pipeline.

Orchestrates: config loading -> teacher setup -> student creation ->
data loading -> training -> evaluation -> saving.
"""

import logging
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import yaml

from ..models.vit_teacher import create_teacher, TeacherWrapper, get_device
from ..models.vit_student import create_student_from_teacher
from ..utils.metrics import print_model_summary, get_compression_metrics, evaluate
from .trainer import NSAViTTrainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_data_loaders(config: dict) -> tuple:
    """
    Create train and validation data loaders.

    Supports CIFAR-100 and ImageNet.
    """
    dataset_name = config.get('dataset', 'cifar100')
    data_root = config.get('data_root', './data')
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    image_size = config.get('image_size', 224)

    if dataset_name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
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

    elif dataset_name == 'imagenet':
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(train_dir,
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir,
                                           transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    logger.info(f"Dataset: {dataset_name}, "
                f"train={len(train_dataset)}, val={len(val_dataset)}")
    return train_loader, val_loader


def run_distillation(config_path: str, overrides: dict = None):
    """
    Full NSA-ViT distillation pipeline.

    Steps:
    1. Load config
    2. Create teacher (timm pretrained or fine-tuned checkpoint)
    3. Create student (low-rank, SVD-initialized from teacher)
    4. Print compression statistics
    5. Set up data loaders
    6. Train with NSA losses
    7. Final evaluation
    """
    # 1. Load config
    config = load_config(config_path)
    if overrides:
        config.update(overrides)

    # Weights & Biases: init run (or use existing run from sweep agent)
    use_wandb = config.get('use_wandb', False)
    wandb_available = False
    if use_wandb:
        try:
            import wandb
            wandb_available = True
            if wandb.run is None:
                wandb.init(
                    project=config.get('wandb_project', 'nsa-vit'),
                    entity=config.get('wandb_entity'),
                    name=config.get('wandb_run_name'),
                    config=config,
                )
            else:
                wandb.config.update(config, allow_val_change=True)
        except ImportError:
            logger.warning("use_wandb is True but wandb not installed; skipping wandb logging.")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(config.get('log_dir', './runs'), 'train.log'),
                mode='w'),
        ]
    )

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = get_device(config.get('device', 'auto'))
    logger.info(f"Device: {device}")

    # 2. Create teacher
    teacher = create_teacher(
        model_name=config['teacher_model'],
        num_classes=config.get('num_classes', 100),
        pretrained=True,
        checkpoint_path=config.get('teacher_checkpoint'),
    ).to(device)

    teacher_wrapper = TeacherWrapper(teacher)

    # 3. Create student
    logger.info("Creating low-rank student model...")
    student = create_student_from_teacher(teacher, config)

    # 4. Print compression statistics
    print_model_summary(teacher, student, device=str(device))
    if use_wandb and wandb_available:
        metrics = get_compression_metrics(teacher, student, device=str(device))
        wandb.run.summary.update(metrics)

    # 5. Data loaders
    train_loader, val_loader = get_data_loaders(config)

    # 6. Train
    trainer = NSAViTTrainer(
        teacher_wrapper=teacher_wrapper,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    epochs = config.get('epochs', 50)
    trainer.train(num_epochs=epochs)

    # 7. Final evaluation
    logger.info("Final evaluation on validation set:")
    final_accs = evaluate(trainer.student, val_loader, device, top_k=(1, 5))
    logger.info(f"Final top-1: {final_accs[1]:.2f}%, "
                f"top-5: {final_accs[5]:.2f}%")

    # Cleanup teacher hooks
    teacher_wrapper.remove_hooks()

    return trainer
