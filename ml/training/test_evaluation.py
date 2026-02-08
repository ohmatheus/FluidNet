from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import project_config
from config.training_config import TrainingConfig
from dataset.npz_sequence import FluidNPZSequenceDataset
from training.metrics import (
    MetricsTracker,
    compute_collider_violation,
    compute_divergence_norm,
    compute_emitter_density_accuracy,
    compute_kinetic_energy,
    compute_per_channel_mse,
    compute_ssim_density,
)


def _compute_batch_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    inputs: torch.Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    density_pred = outputs[:, 0, :, :]
    velx_pred = outputs[:, 1, :, :]
    vely_pred = outputs[:, 2, :, :]
    emitter_mask = inputs[:, 4, :, :]
    collider_mask = inputs[:, 5, :, :]

    metrics: dict[str, float] = {}
    metrics.update(compute_per_channel_mse(outputs, targets))
    metrics["divergence_norm"] = compute_divergence_norm(
        velx_pred, vely_pred,
        dx=config.physics_loss.grid_spacing,
        dy=config.physics_loss.grid_spacing,
        padding_mode=config.padding_mode,
    )
    metrics["kinetic_energy"] = compute_kinetic_energy(velx_pred, vely_pred)
    metrics["collider_violation"] = compute_collider_violation(density_pred, collider_mask)
    metrics["emitter_density_accuracy"] = compute_emitter_density_accuracy(
        density_pred, emitter_mask, expected_injection=0.8
    )
    metrics["ssim_density"] = compute_ssim_density(outputs, targets)

    return metrics


def run_test_evaluation(
    model: nn.Module,
    config: TrainingConfig,
    test_indices: list[int],
    device: str,
) -> dict[str, float]:
    if not test_indices:
        print("No test indices provided, skipping test evaluation.")
        return {}

    npz_dir = config.npz_dir / str(project_config.simulation.grid_resolution)

    print(f"\n{'=' * 70}")
    print("TEST SET EVALUATION (best model + persistence baseline)")
    print(f"{'=' * 70}")
    print(f"Loading test dataset ({len(test_indices)} sequences)...")

    test_ds = FluidNPZSequenceDataset(
        npz_dir=npz_dir,
        normalize=config.normalize,
        seq_indices=test_indices,
        is_training=False,
        augmentation_config=None,
        preload=config.preload_dataset,
        rollout_steps=1,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print(f"Samples: {len(test_ds)} | Device: {device}")

    model_tracker = MetricsTracker()
    persistence_tracker = MetricsTracker()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Test evaluation", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Model prediction
            if config.amp_enabled and device == "cuda":
                with autocast(device_type=device):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            model_metrics = _compute_batch_metrics(outputs, targets, inputs, config)
            model_tracker.update(model_metrics)

            # Persistence baseline: predict current frame = next frame
            persistence_pred = inputs[:, 0:3, :, :]
            persistence_metrics = _compute_batch_metrics(persistence_pred, targets, inputs, config)
            persistence_tracker.update(persistence_metrics)

    model_avg = model_tracker.compute_averages()
    persistence_avg = persistence_tracker.compute_averages()

    # Log to MLflow
    mlflow.log_metrics({f"test_{k}": v for k, v in model_avg.items()})
    mlflow.log_metrics({f"test_persistence_{k}": v for k, v in persistence_avg.items()})

    _print_results_table(model_avg, persistence_avg)

    return {"model": model_avg, "persistence": persistence_avg}


def _print_results_table(model: dict[str, float], persistence: dict[str, float]) -> None:
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Model':>15} {'Persistence':>15}")
    print("-" * 70)

    display_order = [
        ("mse_density", "MSE Density"),
        ("mse_velx", "MSE Vel-X"),
        ("mse_vely", "MSE Vel-Y"),
        ("divergence_norm", "Divergence Norm"),
        ("ssim_density", "SSIM Density"),
        ("collider_violation", "Collider Violation"),
        ("emitter_density_accuracy", "Emitter Accuracy"),
        ("kinetic_energy", "Kinetic Energy"),
    ]

    for key, label in display_order:
        m_val = model.get(key, 0.0)
        p_val = persistence.get(key, 0.0)
        print(f"{label:<25} {m_val:>15.6f} {p_val:>15.6f}")

    print("=" * 70)
