import random
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig, project_config
from dataset.npz_sequence import FluidNPZSequenceDataset
from models.unet import UNet, UNetConfig
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(
    npz_dir: Path, split_ratios: tuple[float, float, float], seed: int
) -> tuple[list[int], list[int], list[int]]:
    seq_paths = sorted([p for p in npz_dir.iterdir() if p.name.startswith("seq_") and p.name.endswith(".npz")])
    n_seq = len(seq_paths)

    if n_seq < 3:
        raise ValueError(
            f"Need at least 3 sequences for train/val/test splits, found {n_seq}. "
            f"Generate more data or reduce number of splits."
        )

    indices = list(range(n_seq))
    random.Random(seed).shuffle(indices)

    # reserve 1 sample per split
    reserved_per_split = 1
    remaining = n_seq - 3

    # normalize ratios (handle edge case where sum != 1.0)
    ratio_sum = sum(split_ratios)
    if ratio_sum == 0:
        normalized_ratios = (1 / 3, 1 / 3, 1 / 3)
    else:
        normalized_ratios = cast("tuple[float, float, float]", tuple(r / ratio_sum for r in split_ratios))

    # floor
    additional = [int(r * remaining) for r in normalized_ratios]

    # distribute leftover samples
    leftover = remaining - sum(additional)
    if leftover > 0:
        fractional_parts = [(r * remaining) % 1 for r in normalized_ratios]
        sorted_indices = sorted(range(3), key=lambda i: fractional_parts[i], reverse=True)
        for i in range(leftover):
            additional[sorted_indices[i]] += 1

    n_train = reserved_per_split + additional[0]
    n_val = reserved_per_split + additional[1]
    n_test = reserved_per_split + additional[2]

    assert n_train + n_val + n_test == n_seq, (
        f"Split sizes don't sum to n_seq: {n_train} + {n_val} + {n_test} != {n_seq}"
    )

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def main() -> None:
    config = TrainingConfig()  # (can add yaml loading + HP tuning later)

    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {config.device}")
    else:
        print(f"Using device: {config.device}")

    set_seed(config.split_seed)

    npz_dir: Path = config.npz_dir / str(project_config.simulation.grid_resolution)

    train_idx, val_idx, test_idx = make_splits(npz_dir, config.split_ratios, config.split_seed)

    print(f"Dataset splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    aug_config_dict = {
        "enable_augmentation": config.augmentation.enable_augmentation,
        "flip_probability": config.augmentation.flip_probability,
    }

    train_ds = FluidNPZSequenceDataset(
        npz_dir=npz_dir,
        normalize=config.normalize,
        seq_indices=train_idx,
        is_training=True,
        augmentation_config=aug_config_dict,
        preload=config.preload_dataset,
        rollout_steps=config.rollout_steps,
    )

    # Validation dataset: K=1 or matches training K
    val_rollout_steps = config.rollout_steps if config.validation_use_rollout_k else 1
    val_ds = FluidNPZSequenceDataset(
        npz_dir=npz_dir,
        normalize=config.normalize,
        seq_indices=val_idx,
        is_training=False,
        augmentation_config=None,
        preload=config.preload_dataset,
        rollout_steps=val_rollout_steps,
    )

    # to compare models later
    # test_ds = FluidNPZSequenceDataset(
    #    npz_dir=npz_dir, normalize=config.normalize, seq_indices=test_idx, fake_empty_pct=config.fake_empty_pct
    # )

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = UNet(
        cfg=UNetConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            depth=config.depth,
            norm=config.norm,
            act=config.act,
            group_norm_groups=config.group_norm_groups,
            dropout=config.dropout,
            upsample=config.upsample,
            use_residual=config.use_residual,
            bottleneck_blocks=config.bottleneck_blocks,
        )
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name="dummy"):
        mlflow.log_params(
            {
                # Model identity
                "model_name": model.__class__.__name__,
                "model_params": total_params,
                # Basic training
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "amp_enabled": config.amp_enabled,
                "device": config.device,
                "num_workers": config.num_workers,
                # Physics loss configuration
                "physics_loss.mse_weight": config.physics_loss.mse_weight,
                "physics_loss.divergence_weight": config.physics_loss.divergence_weight,
                "physics_loss.gradient_weight": config.physics_loss.gradient_weight,
                "physics_loss.emitter_weight": config.physics_loss.emitter_weight,
                "physics_loss.enable_divergence": config.physics_loss.enable_divergence,
                "physics_loss.enable_gradient": config.physics_loss.enable_gradient,
                "physics_loss.enable_emitter": config.physics_loss.enable_emitter,
                "physics_loss.grid_spacing": config.physics_loss.grid_spacing,
                # Gradient clipping
                "gradient_clip_norm": config.gradient_clip_norm,
                "gradient_clip_enabled": config.gradient_clip_enabled,
                # LR scheduler
                "use_lr_scheduler": config.use_lr_scheduler,
                "lr_scheduler_type": config.lr_scheduler_type,
                "lr_scheduler_patience": config.lr_scheduler_patience,
                "lr_scheduler_factor": config.lr_scheduler_factor,
                "lr_scheduler_min_lr": config.lr_scheduler_min_lr,
                "lr_scheduler_step_size": config.lr_scheduler_step_size,
                "lr_scheduler_t_max": config.lr_scheduler_t_max,
                # Early stopping
                "use_early_stopping": config.use_early_stopping,
                "early_stop_patience": config.early_stop_patience,
                "early_stop_min_delta": config.early_stop_min_delta,
                # Dataset configuration
                "normalize": config.normalize,
                "split_ratios": str(config.split_ratios),
                "split_seed": config.split_seed,
                "preload_dataset": config.preload_dataset,
                # Augmentation configuration
                "augmentation.enable": config.augmentation.enable_augmentation,
                "augmentation.flip_probability": config.augmentation.flip_probability,
                "augmentation.flip_axis": config.augmentation.flip_axis,
                # Multi-step rollout training
                "rollout_steps": config.rollout_steps,
                "rollout_schedule": str(config.rollout_schedule),
                "rollout_weight_decay": config.rollout_weight_decay,
                "rollout_gradient_truncation": config.rollout_gradient_truncation,
                "rollout_reset_lr_on_k_change": config.rollout_reset_lr_on_k_change,
                "validation_use_rollout_k": config.validation_use_rollout_k,
                # Model architecture
                "in_channels": config.in_channels,
                "out_channels": config.out_channels,
                "base_channels": config.base_channels,
                "depth": config.depth,
                "norm": config.norm,
                "act": config.act,
                "group_norm_groups": config.group_norm_groups,
                "dropout": config.dropout,
                "upsample": config.upsample,
                "use_residual": config.use_residual,
                "bottleneck_blocks": config.bottleneck_blocks,
                "output_activation": config.output_activation,
                # Checkpoint settings
                "save_every_n_epochs": config.save_every_n_epochs,
                "keep_last_n_checkpoints": config.keep_last_n_checkpoints,
            }
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=config.device,
            train_indices=train_idx,
            val_indices=val_idx,
        )

        trainer.train()

    print("Training complete!")


if __name__ == "__main__":
    # Must be set before any CUDA operations when using DataLoader with num_workers > 0
    mp.set_start_method("spawn", force=True)
    main()
