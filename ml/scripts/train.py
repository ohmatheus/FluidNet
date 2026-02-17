import argparse
import subprocess
import sys
from pathlib import Path

import mlflow
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from config.config import PROJECT_ROOT_PATH
from config.training_config import TrainingConfig, VariantMetadata, project_config
from dataset.npz_sequence import FluidNPZSequenceDataset
from models.unet import UNet, UNetConfig
from scripts.variant_manager import VariantManager
from training.test_evaluation import run_rollout_evaluation, run_test_evaluation
from training.trainer import Trainer
from utils.data_splits import make_splits, set_seed


def dict_to_training_config(config_dict: dict) -> TrainingConfig:
    config_dict.pop("variant_name", None)
    config_dict.pop("model_architecture", None)
    config_dict.pop("parent_variant", None)

    if "augmentation" in config_dict and isinstance(config_dict["augmentation"], dict):
        from config.training_config import AugmentationConfig

        config_dict["augmentation"] = AugmentationConfig(**config_dict["augmentation"])

    if "physics_loss" in config_dict and isinstance(config_dict["physics_loss"], dict):
        from config.training_config import PhysicsLossConfig

        config_dict["physics_loss"] = PhysicsLossConfig(**config_dict["physics_loss"])

    return TrainingConfig(**config_dict)


def train_single_variant(
    variant_name: str,
    manager: VariantManager,
    run_inference: bool = True,
    device: str | None = None,
    parent_run_id: str | None = None,
) -> str:
    print(f"\n{'=' * 80}")
    print(f"Training variant: {variant_name}")
    print(f"{'=' * 80}\n")

    config_dict = manager.build_config_with_inheritance(variant_name)

    variant_meta = config_dict["_variant_metadata"]

    config = dict_to_training_config(config_dict)

    config.variant = VariantMetadata(
        variant_name=variant_name,
        model_architecture_name=variant_meta["model_architecture_name"],
        full_model_name=variant_meta["full_model_name"],
        parent_variant=variant_meta["parent_variant"],
        warmstart_checkpoint=manager.get_warmstart_checkpoint(variant_name),
        variant_yaml_path=manager.get_variant_yaml_path(variant_name),
        relative_dir=variant_meta["relative_dir"],
    )

    if device:
        config.device = device

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

    train_stride = max(1, round(config.rollout_stride * config.rollout_step))

    train_ds = FluidNPZSequenceDataset(
        npz_dir=npz_dir,
        normalize=config.normalize,
        seq_indices=train_idx,
        is_training=True,
        augmentation_config=aug_config_dict,
        preload=config.preload_dataset,
        rollout_steps=config.rollout_step,
        stride=train_stride,
    )

    val_rollout_steps = config.rollout_step if config.validation_use_rollout_k else 1
    val_ds = FluidNPZSequenceDataset(
        npz_dir=npz_dir,
        normalize=config.normalize,
        seq_indices=val_idx,
        is_training=False,
        augmentation_config=None,
        preload=config.preload_dataset,
        rollout_steps=val_rollout_steps,
    )

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
            downsample=config.downsample,
            padding_mode=config.padding_mode,
            use_residual=config.use_residual,
            bottleneck_blocks=config.bottleneck_blocks,
            output_activation=config.output_activation,
        )
    ).to(config.device)

    # Load warmstart weights if available
    if config.variant.warmstart_checkpoint:
        print(f"Warm-start from parent: {config.variant.parent_variant}")
        print(f"Checkpoint: {config.variant.warmstart_checkpoint}")
        checkpoint = torch.load(config.variant.warmstart_checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded parent weights successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.variant.full_model_name) as run:
        run_id: str = run.info.run_id

        mlflow.set_tags(
            {
                "variant_name": variant_name,
                "model_architecture": config.variant.model_architecture_name,
                "full_model_name": config.variant.full_model_name,
                "parent_variant": config.variant.parent_variant or "none",
                "rollout_k": config.rollout_step,
                "relative_dir": str(config.variant.relative_dir),
            }
        )

        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)

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
                "physics_loss.stencil_mode": config.physics_loss.stencil_mode,
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
                "rollout_step": config.rollout_step,
                "rollout_stride": config.rollout_stride,
                "train_stride": train_stride,
                "rollout_weight_decay": config.rollout_weight_decay,
                "rollout_gradient_truncation": config.rollout_gradient_truncation,
                "validation_use_rollout_k": config.validation_use_rollout_k,
                "rollout_final_step_only": config.rollout_final_step_only,
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
                "downsample": config.downsample,
                "padding_mode": config.padding_mode,
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

        if test_idx:
            run_test_evaluation(
                model=model,
                config=config,
                test_indices=test_idx,
                device=config.device,
            )
            run_rollout_evaluation(
                model=model,
                config=config,
                test_indices=test_idx,
                device=config.device,
            )

        print(f"\nTraining complete: {config.variant.full_model_name}")
        print(f"Best model: {config.checkpoint_dir_variant / 'best_model.pth'}")

    if run_inference and config_dict.get("auto_inference_enabled", True):
        run_auto_inference(variant_name, manager)

    return run_id


def run_auto_inference(variant_name: str, manager: VariantManager) -> None:
    print(f"\n{'=' * 80}")
    print(f"Running auto-inference for {variant_name}")
    print(f"{'=' * 80}\n")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "simple_infer.py"),
        "--variant",
        variant_name,
        "--num-frames",
        "600",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Inference complete")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Inference failed: {e}")
        print("Continuing with training pipeline...")
    except FileNotFoundError:
        print("WARNING: simple_infer.py not found or doesn't support --variant yet")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FluidNet with hierarchical multi-variant support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )

    parser.add_argument("variants", nargs="+", help="One or more variant names to train (e.g., K1_A K2_B)")

    parser.add_argument(
        "--no-deps", action="store_true", help="Skip automatic dependency resolution (train only specified variants)"
    )

    parser.add_argument("--force-retrain", action="store_true", help="Force retrain even if checkpoint exists")

    parser.add_argument("--no-inference", action="store_true", help="Skip automatic inference after training")

    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")

    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    ml_root = Path(__file__).parent.parent
    manager = VariantManager(
        config_root=ml_root / "config", checkpoints_dir=Path(PROJECT_ROOT_PATH) / "data" / "checkpoints"
    )

    print(f"Discovered {len(manager._variants_cache)} variants:")
    for name in sorted(manager._variants_cache.keys()):
        print(f"  - {name}")

    # Resolve dependencies
    if args.no_deps:
        variants_to_train = args.variants
        print(f"\nTraining {len(variants_to_train)} variants (no dependency resolution):")
        for v in variants_to_train:
            print(f"  - {v}")
    else:
        variants_to_train = manager.topological_sort(args.variants)
        print("\nDependency resolution:")
        for v in variants_to_train:
            status = "[REQUESTED]" if v in args.variants else "[DEPENDENCY]"
            print(f"  {status} {v}")

    # Filter already-trained
    if not args.force_retrain:
        filtered = []
        for v in variants_to_train:
            is_explicitly_requested = v in args.variants

            if is_explicitly_requested:
                filtered.append(v)
                if manager.check_checkpoint_exists(v):
                    print(f"Retraining {v} (explicitly requested, checkpoint exists)")
            elif manager.check_checkpoint_exists(v):
                print(f"Skipping {v} (dependency checkpoint exists, use --force-retrain to override)")
            else:
                # Train dependencies that don't have checkpoints
                filtered.append(v)
        variants_to_train = filtered

    if not variants_to_train:
        print("\nNo variants to train (all checkpoints exist)")
        return

    print(f"\nWill train {len(variants_to_train)} variants")

    # Train variants in dependency order
    run_ids: dict[str, str] = {}
    for variant_name in variants_to_train:
        parent_variant = manager.get_parent_variant(variant_name)
        parent_run_id = run_ids.get(parent_variant) if parent_variant else None

        run_id = train_single_variant(
            variant_name=variant_name,
            manager=manager,
            run_inference=not args.no_inference,
            device=args.device,
            parent_run_id=parent_run_id,
        )

        run_ids[variant_name] = run_id

    print(f"\n{'=' * 80}")
    print("All training complete!")
    print(f"{'=' * 80}")
    print(f"Trained {len(variants_to_train)} variants:")
    for v in variants_to_train:
        print(f"  - {v}")


if __name__ == "__main__":
    # Must be set before any CUDA operations when using DataLoader with num_workers > 0
    mp.set_start_method("spawn", force=True)
    main()
