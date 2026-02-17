import shutil
import tempfile
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import PROJECT_ROOT_PATH, project_config
from config.training_config import TrainingConfig
from dataset.normalization import load_normalization_scales
from dataset.npz_sequence import FluidNPZSequenceDataset
from training.physics_loss import StencilMode
from training.metrics import (
    MetricsTracker,
    compute_collider_violation,
    compute_divergence_norm,
    compute_emitter_density_accuracy,
    compute_gradient_l1,
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
        velx_pred,
        vely_pred,
        dx=config.physics_loss.grid_spacing,
        dy=config.physics_loss.grid_spacing,
        padding_mode=config.padding_mode,
        mode=cast(StencilMode, config.physics_loss.stencil_mode),
    )
    metrics["kinetic_energy"] = compute_kinetic_energy(velx_pred, vely_pred)
    metrics["collider_violation"] = compute_collider_violation(density_pred, collider_mask)
    metrics["emitter_density_accuracy"] = compute_emitter_density_accuracy(
        density_pred, emitter_mask, expected_injection=0.8
    )
    metrics["ssim_density"] = compute_ssim_density(outputs, targets)
    metrics["gradient_l1"] = compute_gradient_l1(
        density_pred,
        targets[:, 0, :, :],
        dx=config.physics_loss.grid_spacing,
        dy=config.physics_loss.grid_spacing,
        padding_mode=config.padding_mode,
        mode=cast(StencilMode, config.physics_loss.stencil_mode),
    )

    return metrics


def run_test_evaluation(
    model: nn.Module,
    config: TrainingConfig,
    test_indices: list[int],
    device: str,
) -> dict[str, dict[str, float]]:
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
        ("gradient_l1", "Gradient L1"),
        ("collider_violation", "Collider Violation"),
        ("emitter_density_accuracy", "Emitter Accuracy"),
        ("kinetic_energy", "Kinetic Energy"),
    ]

    for key, label in display_order:
        m_val = model.get(key, 0.0)
        p_val = persistence.get(key, 0.0)
        print(f"{label:<25} {m_val:>15.6f} {p_val:>15.6f}")

    print("=" * 70)


ROLLOUT_STEPS = 30
STARTING_POINTS: list[int | str] = [20, "middle", -31]


def _get_starting_frames(total_frames: int) -> list[int]:
    starts = []
    for sp in STARTING_POINTS:
        if sp == "middle":
            t = total_frames // 2
        elif isinstance(sp, int) and sp < 0:
            t = total_frames + sp
        else:
            t = int(sp)

        if t >= 1 and t + ROLLOUT_STEPS < total_frames:
            starts.append(t)
    return starts


def _run_single_rollout(
    model: nn.Module,
    data: dict[str, np.ndarray],
    t_start: int,
    norm_scales: dict[str, float] | None,
    device: str,
    use_amp: bool,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    d = data["density"]
    vx = data["velx"]
    vz = data["velz"]
    emitter = data["emitter"]
    collider = data["collider"]

    def norm(arr: np.ndarray, key: str) -> np.ndarray:
        if norm_scales:
            return arr / norm_scales[key]
        return arr

    # Initial state
    d_t = norm(d[t_start], "S_density")
    d_tminus = norm(d[t_start - 1], "S_density")
    vx_t = norm(vx[t_start], "S_velx")
    vz_t = norm(vz[t_start], "S_velz")

    state_current = torch.tensor(np.stack([d_t, vx_t, vz_t], axis=0), dtype=torch.float32, device=device).unsqueeze(0)
    state_prev = torch.tensor(d_tminus[np.newaxis], dtype=torch.float32, device=device).unsqueeze(0)

    step_metrics: list[dict[str, float]] = []

    for k in range(ROLLOUT_STEPS):
        t_cur = t_start + k
        t_next = t_start + k + 1

        emitter_t = torch.tensor(emitter[t_cur][np.newaxis], dtype=torch.float32, device=device).unsqueeze(0)
        collider_t = torch.tensor(collider[t_cur][np.newaxis], dtype=torch.float32, device=device).unsqueeze(0)

        model_input = torch.cat([state_current, state_prev, emitter_t, collider_t], dim=1)

        if use_amp and device == "cuda":
            with autocast(device_type=device):
                pred = model(model_input)
        else:
            pred = model(model_input)

        # GT for this step
        gt_d = norm(d[t_next], "S_density")
        gt_vx = norm(vx[t_next], "S_velx")
        gt_vz = norm(vz[t_next], "S_velz")
        gt = torch.tensor(np.stack([gt_d, gt_vx, gt_vz], axis=0), dtype=torch.float32, device=device).unsqueeze(0)

        mse_density = torch.mean((pred[:, 0] - gt[:, 0]) ** 2).item()
        mse_velx = torch.mean((pred[:, 1] - gt[:, 1]) ** 2).item()
        mse_vely = torch.mean((pred[:, 2] - gt[:, 2]) ** 2).item()
        ssim_density = compute_ssim_density(pred, gt)

        divergence_norm = compute_divergence_norm(
            pred[:, 1, :, :],
            pred[:, 2, :, :],
            dx=config.physics_loss.grid_spacing,
            dy=config.physics_loss.grid_spacing,
            padding_mode=config.padding_mode,
            mode=cast(StencilMode, config.physics_loss.stencil_mode),
        )

        divergence_norm_gt = compute_divergence_norm(
            gt[:, 1, :, :],
            gt[:, 2, :, :],
            dx=config.physics_loss.grid_spacing,
            dy=config.physics_loss.grid_spacing,
            padding_mode=config.padding_mode,
            mode=cast(StencilMode, config.physics_loss.stencil_mode),
        )

        gradient_l1 = compute_gradient_l1(
            pred[:, 0, :, :],
            gt[:, 0, :, :],
            dx=config.physics_loss.grid_spacing,
            dy=config.physics_loss.grid_spacing,
            padding_mode=config.padding_mode,
            mode=cast(StencilMode, config.physics_loss.stencil_mode),
        )

        step_metrics.append(
            {
                "mse_density": mse_density,
                "mse_velx": mse_velx,
                "mse_vely": mse_vely,
                "ssim_density": ssim_density,
                "divergence_norm": divergence_norm,
                "divergence_norm_gt": divergence_norm_gt,
                "gradient_l1": gradient_l1,
            }
        )

        # Autoregressive update
        state_prev = state_current[:, 0:1, :, :]
        state_current = pred

    return step_metrics


def _plot_rollout_degradation(avg_per_step: list[dict[str, float]]) -> Figure:
    steps = list(range(1, len(avg_per_step) + 1))
    mse_d = [s["mse_density"] for s in avg_per_step]
    mse_vx = [s["mse_velx"] for s in avg_per_step]
    mse_vy = [s["mse_vely"] for s in avg_per_step]
    ssim_d = [s["ssim_density"] for s in avg_per_step]
    div_norm = [s["divergence_norm"] for s in avg_per_step]
    grad_l1 = [s["gradient_l1"] for s in avg_per_step]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: MSE + SSIM
    ax_mse = ax_left
    ax_mse.plot(steps, mse_d, color="blue", linewidth=2, label="MSE Density")
    ax_mse.plot(steps, mse_vx, color="orange", linewidth=2, label="MSE Vel-X")
    ax_mse.plot(steps, mse_vy, color="green", linewidth=2, label="MSE Vel-Y")
    ax_mse.set_xlabel("Autoregressive Step")
    ax_mse.set_ylabel("MSE", color="blue")
    ax_mse.tick_params(axis="y", labelcolor="blue")
    ax_mse.grid(True, alpha=0.3)

    ax_ssim = ax_mse.twinx()
    ax_ssim.plot(steps, ssim_d, color="teal", linewidth=2, linestyle="--", label="SSIM Density")
    ax_ssim.set_ylabel("SSIM", color="teal")
    ax_ssim.tick_params(axis="y", labelcolor="teal")
    ax_ssim.set_ylim([0, 1])

    lines1, labels1 = ax_mse.get_legend_handles_labels()
    lines2, labels2 = ax_ssim.get_legend_handles_labels()
    ax_mse.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax_mse.set_title("MSE per Channel + SSIM")

    # Right: Divergence + Gradient
    ax_div = ax_right
    ax_div.plot(steps, div_norm, color="red", linewidth=2, label="Divergence Norm")
    ax_div.set_xlabel("Autoregressive Step")
    ax_div.set_ylabel("Divergence Norm", color="red")
    ax_div.tick_params(axis="y", labelcolor="red")
    ax_div.grid(True, alpha=0.3)

    ax_grad = ax_div.twinx()
    ax_grad.plot(steps, grad_l1, color="brown", linewidth=2, linestyle="--", label="Gradient L1")
    ax_grad.set_ylabel("Gradient L1", color="brown")
    ax_grad.tick_params(axis="y", labelcolor="brown")

    lines1, labels1 = ax_div.get_legend_handles_labels()
    lines2, labels2 = ax_grad.get_legend_handles_labels()
    ax_div.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax_div.set_title("Divergence Norm + Gradient L1")

    plt.suptitle("Rollout Degradation (30 steps)", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2.0)

    return fig


def log_artifact_flat(fig: Figure, filename: str, dpi: int = 72, artifact_path: str = "plots") -> None:
    tmp_dir = tempfile.mkdtemp()
    try:
        path = Path(tmp_dir) / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    finally:
        shutil.rmtree(tmp_dir)
        plt.close(fig)


def run_rollout_evaluation(
    model: nn.Module,
    config: TrainingConfig,
    test_indices: list[int],
    device: str,
) -> dict[str, list[dict[str, float]]]:
    if not test_indices:
        print("No test indices provided, skipping rollout evaluation.")
        return {}

    npz_dir = config.npz_dir / str(project_config.simulation.grid_resolution)
    all_seq_paths = sorted([p for p in npz_dir.iterdir() if p.name.startswith("seq_") and p.name.endswith(".npz")])
    test_paths = [all_seq_paths[i] for i in test_indices]

    norm_scales = None
    if config.normalize:
        stats_path = PROJECT_ROOT_PATH / project_config.vdb_tools.stats_output_file
        norm_scales = load_normalization_scales(stats_path)

    print(f"\n{'=' * 70}")
    print("ROLLOUT DEGRADATION EVALUATION")
    print(f"{'=' * 70}")
    print(f"Sequences: {len(test_paths)} | Steps: {ROLLOUT_STEPS} | Starting points: {len(STARTING_POINTS)}")

    # Collect all rollout results: [step][rollout_idx] -> metrics
    all_rollouts: list[list[dict[str, float]]] = [[] for _ in range(ROLLOUT_STEPS)]
    total_rollouts = 0

    model.eval()
    with torch.no_grad():
        for seq_path in tqdm(test_paths, desc="Rollout evaluation", leave=False):
            with np.load(seq_path) as npz:
                data = {
                    "density": npz["density"].astype(np.float32),
                    "velx": npz["velx"].astype(np.float32),
                    "velz": npz["velz"].astype(np.float32),
                    "emitter": npz.get("emitter", np.zeros_like(npz["density"])).astype(np.float32),
                    "collider": npz.get("collider", np.zeros_like(npz["density"])).astype(np.float32),
                }

            T = data["density"].shape[0]
            starting_frames = _get_starting_frames(T)

            for t_start in starting_frames:
                step_metrics = _run_single_rollout(
                    model, data, t_start, norm_scales, device, config.amp_enabled, config
                )
                for k, m in enumerate(step_metrics):
                    all_rollouts[k].append(m)
                total_rollouts += 1

    # Average per step
    avg_per_step: list[dict[str, float]] = []
    for k in range(ROLLOUT_STEPS):
        if not all_rollouts[k]:
            avg_per_step.append(
                {
                    "mse_density": 0.0,
                    "mse_velx": 0.0,
                    "mse_vely": 0.0,
                    "ssim_density": 0.0,
                    "divergence_norm": 0.0,
                    "divergence_norm_gt": 0.0,
                    "gradient_l1": 0.0,
                }
            )
            continue
        avg = {
            key: sum(r[key] for r in all_rollouts[k]) / len(all_rollouts[k])
            for key in [
                "mse_density",
                "mse_velx",
                "mse_vely",
                "ssim_density",
                "divergence_norm",
                "divergence_norm_gt",
                "gradient_l1",
            ]
        }
        avg_per_step.append(avg)

    # Log scalars to MLflow
    mse_at_30 = avg_per_step[-1]["mse_density"]
    ssim_at_30 = avg_per_step[-1]["ssim_density"]
    div_at_30 = avg_per_step[-1]["divergence_norm"]
    grad_at_30 = avg_per_step[-1]["gradient_l1"]
    mlflow.log_metric("test_rollout_mse_density_step30", mse_at_30)
    mlflow.log_metric("test_rollout_ssim_density_step30", ssim_at_30)
    mlflow.log_metric("test_rollout_divergence_norm_step30", div_at_30)
    mlflow.log_metric("test_rollout_gradient_l1_step30", grad_at_30)

    # Plot and log
    fig = _plot_rollout_degradation(avg_per_step)
    log_artifact_flat(fig, "rollout_degradation.png", dpi=100)

    # Console output
    print(f"\nTotal rollouts: {total_rollouts} ({len(test_paths)} sequences x {len(STARTING_POINTS)} starts)")
    print(
        f"\n{'Step':<8} {'MSE Density':>12} {'MSE Vel-X':>12} {'MSE Vel-Y':>12} "
        f"{'SSIM':>10} {'Div Pred':>12} {'Div GT':>12} {'Grad L1':>12}"
    )
    print("-" * 100)
    for k in [0, 4, 9, 14, 19, 24, 29]:
        if k < len(avg_per_step):
            s = avg_per_step[k]
            print(
                f"{k + 1:<8} {s['mse_density']:>12.6f} {s['mse_velx']:>12.6f} {s['mse_vely']:>12.6f} "
                f"{s['ssim_density']:>10.4f} {s['divergence_norm']:>12.6f} {s['divergence_norm_gt']:>12.6f} "
                f"{s['gradient_l1']:>12.6f}"
            )
    print("=" * 100)

    return {"avg_per_step": avg_per_step}
