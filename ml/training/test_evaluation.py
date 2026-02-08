import shutil
import tempfile
from pathlib import Path

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
        velx_pred,
        vely_pred,
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
        ("collider_violation", "Collider Violation"),
        ("emitter_density_accuracy", "Emitter Accuracy"),
        ("kinetic_energy", "Kinetic Energy"),
    ]

    for key, label in display_order:
        m_val = model.get(key, 0.0)
        p_val = persistence.get(key, 0.0)
        print(f"{label:<25} {m_val:>15.6f} {p_val:>15.6f}")

    print("=" * 70)


ROLLOUT_STEPS = 20
STARTING_POINTS: list[int | str] = [20, "middle", -21]


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

        step_metrics.append(
            {
                "mse_density": mse_density,
                "mse_velx": mse_velx,
                "mse_vely": mse_vely,
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, mse_d, color="blue", linewidth=2, label="Density")
    ax.plot(steps, mse_vx, color="orange", linewidth=2, label="Vel-X")
    ax.plot(steps, mse_vy, color="green", linewidth=2, label="Vel-Y")

    ax.set_xlabel("Autoregressive Step")
    ax.set_ylabel("MSE")
    ax.set_title("Rollout Degradation (20 steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

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
                step_metrics = _run_single_rollout(model, data, t_start, norm_scales, device, config.amp_enabled)
                for k, m in enumerate(step_metrics):
                    all_rollouts[k].append(m)
                total_rollouts += 1

    # Average per step
    avg_per_step: list[dict[str, float]] = []
    for k in range(ROLLOUT_STEPS):
        if not all_rollouts[k]:
            avg_per_step.append({"mse_density": 0.0, "mse_velx": 0.0, "mse_vely": 0.0})
            continue
        avg = {
            key: sum(r[key] for r in all_rollouts[k]) / len(all_rollouts[k])
            for key in ["mse_density", "mse_velx", "mse_vely"]
        }
        avg_per_step.append(avg)

    # Log scalar to MLflow
    mse_at_20 = avg_per_step[-1]["mse_density"]
    mlflow.log_metric("test_rollout_mse_density_step20", mse_at_20)

    # Plot and log
    fig = _plot_rollout_degradation(avg_per_step)
    log_artifact_flat(fig, "rollout_degradation.png", dpi=144)

    # Console output
    print(f"\nTotal rollouts: {total_rollouts} ({len(test_paths)} sequences x {len(STARTING_POINTS)} starts)")
    print(f"\n{'Step':<8} {'MSE Density':>15} {'MSE Vel-X':>15} {'MSE Vel-Y':>15}")
    print("-" * 55)
    for k in [0, 4, 9, 14, 19]:
        if k < len(avg_per_step):
            s = avg_per_step[k]
            print(f"{k + 1:<8} {s['mse_density']:>15.6f} {s['mse_velx']:>15.6f} {s['mse_vely']:>15.6f}")
    print("=" * 70)

    return {"avg_per_step": avg_per_step}
