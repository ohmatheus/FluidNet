from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_training_history(history: dict[str, list[float]], best_epoch: int) -> Figure:
    num_epochs = len(history["train_total"])
    if num_epochs == 0:
        raise ValueError("No training history to plot - training never started or was interrupted at epoch 0")

    epochs = list(range(1, num_epochs + 1))

    fig, ax_loss = plt.subplots(1, 1, figsize=(8, 5))

    ax_loss.plot(epochs, history["train_total"], color="#0066CC", linewidth=2.0, label="Train Total", alpha=0.9)
    ax_loss.plot(epochs, history["val_total"], color="#FF5500", linewidth=2.0, label="Val Total", alpha=0.9)
    ax_loss.axvline(
        x=best_epoch, color="red", linestyle="--", linewidth=2.0, label=f"Best Epoch ({best_epoch})", alpha=0.7
    )

    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12, color="#333333")
    ax_loss.set_title("Training and Validation Losses with Learning Rate", fontsize=14, fontweight="bold")
    ax_loss.tick_params(axis="y", labelcolor="#333333")
    ax_loss.grid(True, alpha=0.3)

    loss_range = max(history["train_total"]) / (min(history["train_total"]) + 1e-8)
    if loss_range > 100:
        ax_loss.set_yscale("log")

    ax_lr = ax_loss.twinx()
    ax_lr.plot(epochs, history["learning_rate"], color="#2ca02c", linewidth=2.0, label="Learning Rate", linestyle="--")
    ax_lr.set_ylabel("Learning Rate", fontsize=12, color="#2ca02c")
    ax_lr.tick_params(axis="y", labelcolor="#2ca02c")
    ax_lr.set_yscale("log")

    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10, framealpha=0.9)
    ax_loss.set_xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))

    plt.tight_layout()
    return fig


def plot_loss_components(history: dict[str, list[float]], best_epoch: int) -> Figure:
    num_epochs = len(history["train_total"])
    if num_epochs == 0:
        raise ValueError("No training history to plot")

    epochs = list(range(1, num_epochs + 1))

    components = []
    if history.get("train_mse"):
        components.append(("MSE", "train_mse", "val_mse"))
    if history.get("train_divergence"):
        components.append(("Divergence", "train_divergence", "val_divergence"))
    if history.get("train_gradient"):
        components.append(("Gradient", "train_gradient", "val_gradient"))
    if history.get("train_emitter"):
        components.append(("Emitter", "train_emitter", "val_emitter"))

    n_components = len(components)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (name, train_key, val_key) in enumerate(components):
        ax = axes[idx]
        ax.plot(epochs, history[train_key], color="#0066CC", linewidth=2.0, label="Train", alpha=0.9)
        ax.plot(epochs, history[val_key], color="#FF5500", linewidth=2.0, label="Val", alpha=0.9)
        ax.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(f"{name} Loss", fontsize=10)
        ax.set_title(f"{name} Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        loss_range = max(history[train_key]) / (min(history[train_key]) + 1e-8)
        if loss_range > 100:
            ax.set_yscale("log")

    for idx in range(n_components, 4):
        axes[idx].axis("off")

    plt.suptitle("Loss Components Over Epochs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_metrics_grid(history: dict[str, list[float]]) -> Figure:
    num_epochs = len(history["val_total"])
    if num_epochs == 0:
        raise ValueError("No history to plot")

    epochs = list(range(1, num_epochs + 1))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    axes[0].plot(epochs, history["val_mse_density"], label="Density", linewidth=2)
    axes[0].plot(epochs, history["val_mse_velx"], label="Vel-X", linewidth=2)
    axes[0].plot(epochs, history["val_mse_vely"], label="Vel-Y", linewidth=2)
    axes[0].set_title("Per-Channel MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_divergence_norm"], color="red", linewidth=2)
    axes[1].axhline(y=0.1, color="green", linestyle="--", label="Target < 0.1")
    axes[1].set_title("Divergence Norm")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("||∇·v||₂")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_kinetic_energy"], color="blue", linewidth=2)
    axes[2].set_title("Kinetic Energy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KE (trend monitor)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, history["val_collider_violation"], color="orange", linewidth=2)
    axes[3].axhline(y=0.01, color="green", linestyle="--", label="Target < 0.01")
    axes[3].set_title("Collider Violation")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Density in collider")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(epochs, history["val_emitter_accuracy"], color="purple", linewidth=2)
    axes[4].axhline(y=0.1, color="green", linestyle="--", label="Target < 0.1")
    axes[4].set_title("Emitter Density Accuracy")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("Injection error")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(epochs, history["val_ssim_density"], color="teal", linewidth=2)
    axes[5].axhline(y=0.9, color="green", linestyle="--", label="Target > 0.9")
    axes[5].set_title("SSIM (Density)")
    axes[5].set_xlabel("Epoch")
    axes[5].set_ylabel("SSIM")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(epochs, history["val_gradient_l1"], color="brown", linewidth=2)
    axes[6].set_title("Gradient L1 (Edge Sharpness)")
    axes[6].set_xlabel("Epoch")
    axes[6].set_ylabel("L1 gradient error")
    axes[6].grid(True, alpha=0.3)

    axes[7].axis("off")

    mse_range = max(history["val_mse_density"]) / (min(history["val_mse_density"]) + 1e-8)
    if mse_range > 100:
        axes[0].set_yscale("log")

    plt.suptitle("Validation Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig
