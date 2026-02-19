import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import PROJECT_ROOT_PATH, project_config
from config.training_config import TrainingConfig
from dataset.npz_sequence import FluidNPZSequenceDataset
from training.early_stopping import EarlyStopping
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
from training.physics_loss import PhysicsAwareLoss, StencilMode
from training.test_evaluation import log_artifact_flat


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config.model_copy()
        self.device = device

        self.gradient_clip_norm = config.gradient_clip_norm
        self.gradient_clip_enabled = config.gradient_clip_enabled

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = PhysicsAwareLoss(
            mse_weight=config.physics_loss.mse_weight,
            divergence_weight=config.physics_loss.divergence_weight,
            gradient_weight=config.physics_loss.gradient_weight,
            emitter_weight=config.physics_loss.emitter_weight,
            grid_spacing=config.physics_loss.grid_spacing,
            enable_divergence=config.physics_loss.enable_divergence,
            enable_gradient=config.physics_loss.enable_gradient,
            enable_emitter=config.physics_loss.enable_emitter,
            padding_mode=config.padding_mode,
            stencil_mode=cast("StencilMode", config.physics_loss.stencil_mode),
        )

        self.scaler = GradScaler("cuda") if config.amp_enabled and device == "cuda" else None

        self.scheduler = self._create_scheduler() if config.use_lr_scheduler else None

        # Use variant-specific checkpoint directory if variant is configured
        if config.variant:
            checkpoint_dir = config.checkpoint_dir_variant
        else:
            checkpoint_dir = config.checkpoint_dir / str(model.model_name)

        self.early_stopping = (
            EarlyStopping(
                patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta,
                checkpoint_dir=checkpoint_dir,
            )
            if config.use_early_stopping
            else None
        )

        self.config.checkpoint_dir = checkpoint_dir
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_total": [],
            "train_mse": [],
            "val_total": [],
            "val_mse": [],
            "learning_rate": [],
            "val_mse_density": [],
            "val_mse_velx": [],
            "val_mse_vely": [],
            "val_divergence_norm": [],
            "val_kinetic_energy": [],
            "val_collider_violation": [],
            "val_emitter_accuracy": [],
            "val_ssim_density": [],
            "val_gradient_l1": [],
        }

        if self.config.physics_loss.enable_divergence:
            self.history["train_divergence"] = []
            self.history["val_divergence"] = []

        if self.config.physics_loss.enable_gradient:
            self.history["train_gradient"] = []
            self.history["val_gradient"] = []

        if self.config.physics_loss.enable_emitter:
            self.history["train_emitter"] = []
            self.history["val_emitter"] = []

    def _create_scheduler(self) -> Any:
        if self.config.lr_scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                min_lr=self.config.lr_scheduler_min_lr,
            )
        elif self.config.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_scheduler_t_max,
                eta_min=self.config.lr_scheduler_min_lr,
            )
        elif self.config.lr_scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.lr_scheduler_step_size,
                gamma=self.config.lr_scheduler_factor,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        loss_accumulators = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_data in pbar:
            if len(batch_data) == 4:
                # Rollout mode: (x_0, y_seq, masks, cond)
                inputs, targets, masks, cond = batch_data
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                cond = cond.to(self.device, non_blocking=True)
                rollout_mode = True
            else:
                # Single-step mode: (x, y, cond)
                inputs, targets, cond = batch_data
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                cond = cond.to(self.device, non_blocking=True)
                rollout_mode = False

            self.optimizer.zero_grad()

            # AMP
            if self.scaler is not None:
                with autocast(device_type=self.device):
                    if rollout_mode:
                        loss, loss_dict, _ = self._compute_rollout_loss(inputs, targets, masks, cond)
                    else:
                        outputs = self.model(inputs, cond)
                        loss, loss_dict = self.criterion(outputs, targets, inputs)

                self.scaler.scale(loss).backward()

                if self.gradient_clip_enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if rollout_mode:
                    loss, loss_dict, _ = self._compute_rollout_loss(inputs, targets, masks, cond)
                else:
                    outputs = self.model(inputs, cond)
                    loss, loss_dict = self.criterion(outputs, targets, inputs)

                loss.backward()

                if self.gradient_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

                self.optimizer.step()

            num_batches += 1

            for key, value in loss_dict.items():
                if key not in loss_accumulators:
                    loss_accumulators[key] = 0.0
                loss_accumulators[key] += value

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Average all losses
        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        return avg_losses

    def validate(self) -> dict[str, float]:
        self.model.eval()
        loss_accumulators = {}
        metrics_tracker = MetricsTracker()
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch_data in pbar:
                if len(batch_data) == 4:
                    # Rollout mode: (x_0, y_seq, masks, cond)
                    inputs, targets, masks, cond = batch_data
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    cond = cond.to(self.device, non_blocking=True)
                    rollout_mode = True
                else:
                    # Single-step mode: (x, y, cond)
                    inputs, targets, cond = batch_data
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    cond = cond.to(self.device, non_blocking=True)
                    rollout_mode = False

                # Compute loss (with or without AMP)
                if self.scaler is not None:
                    with autocast(device_type=self.device):
                        if rollout_mode:
                            loss, loss_dict, outputs = self._compute_rollout_loss(inputs, targets, masks, cond)
                        else:
                            outputs = self.model(inputs, cond)
                            loss, loss_dict = self.criterion(outputs, targets, inputs)
                else:
                    if rollout_mode:
                        loss, loss_dict, outputs = self._compute_rollout_loss(inputs, targets, masks, cond)
                    else:
                        outputs = self.model(inputs, cond)
                        loss, loss_dict = self.criterion(outputs, targets, inputs)

                num_batches += 1

                for key, value in loss_dict.items():
                    if key not in loss_accumulators:
                        loss_accumulators[key] = 0.0
                    loss_accumulators[key] += value

                if outputs is not None:
                    density_pred = outputs[:, 0, :, :]
                    velx_pred = outputs[:, 1, :, :]
                    vely_pred = outputs[:, 2, :, :]

                    if rollout_mode:
                        target_for_metrics = targets[:, -1, :, :, :]
                        emitter_mask = masks[:, -1, 0, :, :]
                        collider_mask = masks[:, -1, 1, :, :]
                    else:
                        target_for_metrics = targets
                        emitter_mask = inputs[:, 4, :, :]
                        collider_mask = inputs[:, 5, :, :]

                    batch_metrics = {}

                    batch_metrics.update(compute_per_channel_mse(outputs, target_for_metrics))

                    batch_metrics["divergence_norm"] = compute_divergence_norm(
                        velx_pred,
                        vely_pred,
                        dx=self.config.physics_loss.grid_spacing,
                        dy=self.config.physics_loss.grid_spacing,
                        padding_mode=self.config.padding_mode,
                        mode=cast("StencilMode", self.config.physics_loss.stencil_mode),
                    )

                    batch_metrics["kinetic_energy"] = compute_kinetic_energy(velx_pred, vely_pred)

                    batch_metrics["collider_violation"] = compute_collider_violation(density_pred, collider_mask)

                    batch_metrics["emitter_density_accuracy"] = compute_emitter_density_accuracy(
                        density_pred, emitter_mask, expected_injection=0.8
                    )

                    batch_metrics["ssim_density"] = compute_ssim_density(outputs, target_for_metrics)

                    batch_metrics["gradient_l1"] = compute_gradient_l1(
                        density_pred,
                        target_for_metrics[:, 0, :, :],
                        dx=self.config.physics_loss.grid_spacing,
                        dy=self.config.physics_loss.grid_spacing,
                        padding_mode=self.config.padding_mode,
                        mode=cast("StencilMode", self.config.physics_loss.stencil_mode),
                    )

                    metrics_tracker.update(batch_metrics)

                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        avg_metrics = metrics_tracker.compute_averages()

        return {**avg_losses, **avg_metrics}

    def _compute_rollout_loss(
        self,
        x_0: torch.Tensor,
        y_seq: torch.Tensor,
        masks: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        """
        Compute loss for K-step autoregressive rollout.
        Returns: (loss, loss_dict, final_output)
        """
        K = y_seq.shape[1]

        state_current = x_0[:, :3, :, :]  # [d_t, vx_t, vz_t]
        state_prev = x_0[:, 3:4, :, :]  # [d_t-1]

        if self.config.rollout_final_step_only:
            for k in range(K):
                emitter_k = masks[:, k, 0:1, :, :]
                collider_k = masks[:, k, 1:2, :, :]
                model_input = torch.cat([state_current, state_prev, emitter_k, collider_k], dim=1)

                pred = self.model(model_input, cond)  # (B, 3, H, W)

                if k == K - 1:
                    target_final = y_seq[:, k, :, :, :]
                    final_loss, loss_dict = self.criterion(pred, target_final, model_input)
                    return final_loss, loss_dict, pred

                if self.config.rollout_gradient_truncation:
                    state_prev = pred[:, 0:1, :, :].detach()
                    state_current = pred.detach()
                else:
                    state_prev = pred[:, 0:1, :, :]
                    state_current = pred

            raise RuntimeError("Should have returned on final step")

        else:
            # WEIGHTED-AVERAGE MODE
            total_loss = torch.tensor(0.0, device=x_0.device)
            loss_accumulator: dict[str, float] = {}
            weight_sum = 0.0
            final_pred: torch.Tensor

            for k in range(K):
                emitter_k = masks[:, k, 0:1, :, :]
                collider_k = masks[:, k, 1:2, :, :]
                model_input = torch.cat([state_current, state_prev, emitter_k, collider_k], dim=1)

                pred = self.model(model_input, cond)  # (B, 3, H, W)

                target_k = y_seq[:, k, :, :, :]
                loss_k, loss_dict_k = self.criterion(pred, target_k, model_input)

                weight = self.config.rollout_weight_decay**k
                total_loss += weight * loss_k
                weight_sum += weight

                for key, val in loss_dict_k.items():
                    if key not in loss_accumulator:
                        loss_accumulator[key] = 0.0
                    loss_accumulator[key] += weight * val

                if self.config.rollout_gradient_truncation:
                    state_prev = pred[:, 0:1, :, :].detach()
                    state_current = pred.detach()
                else:
                    state_prev = pred[:, 0:1, :, :]
                    state_current = pred

                final_pred = pred

            total_loss = total_loss / weight_sum
            averaged_dict = {k: v / weight_sum for k, v in loss_accumulator.items()}
            return total_loss, averaged_dict, final_pred

    def _create_dataloader_with_rollout_steps(self, K: int, is_training: bool) -> DataLoader:
        split = "train" if is_training else "val"

        npz_dir = (
            PROJECT_ROOT_PATH
            / project_config.vdb_tools.npz_output_directory
            / str(project_config.simulation.grid_resolution)
        )

        dataset = FluidNPZSequenceDataset(
            npz_dir=npz_dir,
            split=split,
            normalize=self.config.normalize,
            is_training=is_training,
            augmentation_config=self.config.augmentation.model_dump() if is_training else None,
            preload=self.config.preload_dataset,
            rollout_steps=K,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

    def plot_training_history(self) -> Figure:
        num_epochs = len(self.history["train_total"])

        if num_epochs == 0:
            raise ValueError("No training history to plot - training never started or was interrupted at epoch 0")

        epochs = list(range(1, num_epochs + 1))

        if self.early_stopping is not None and self.early_stopping.best_epoch >= 0:
            best_epoch = self.early_stopping.best_epoch + 1
        else:
            min_val_idx = self.history["val_total"].index(min(self.history["val_total"]))
            best_epoch = min_val_idx + 1

        fig, ax_loss = plt.subplots(1, 1, figsize=(8, 5))

        ax_loss.plot(
            epochs,
            self.history["train_total"],
            color="#0066CC",
            linewidth=2.0,
            label="Train Total",
            alpha=0.9,
        )
        ax_loss.plot(
            epochs,
            self.history["val_total"],
            color="#FF5500",
            linewidth=2.0,
            label="Val Total",
            alpha=0.9,
        )

        ax_loss.axvline(
            x=best_epoch, color="red", linestyle="--", linewidth=2.0, label=f"Best Epoch ({best_epoch})", alpha=0.7
        )

        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("Loss", fontsize=12, color="#333333")
        ax_loss.set_title("Training and Validation Losses with Learning Rate", fontsize=14, fontweight="bold")
        ax_loss.tick_params(axis="y", labelcolor="#333333")
        ax_loss.grid(True, alpha=0.3)

        loss_range = max(self.history["train_total"]) / (min(self.history["train_total"]) + 1e-8)
        if loss_range > 100:
            ax_loss.set_yscale("log")

        ax_lr = ax_loss.twinx()
        ax_lr.plot(
            epochs, self.history["learning_rate"], color="#2ca02c", linewidth=2.0, label="Learning Rate", linestyle="--"
        )
        ax_lr.set_ylabel("Learning Rate", fontsize=12, color="#2ca02c")
        ax_lr.tick_params(axis="y", labelcolor="#2ca02c")
        ax_lr.set_yscale("log")

        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_lr.get_legend_handles_labels()
        ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10, framealpha=0.9)

        ax_loss.set_xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))

        plt.tight_layout()

        return fig

    def plot_loss_components(self) -> Figure:
        num_epochs = len(self.history["train_total"])
        if num_epochs == 0:
            raise ValueError("No training history to plot")

        epochs = list(range(1, num_epochs + 1))

        if self.early_stopping is not None and self.early_stopping.best_epoch >= 0:
            best_epoch = self.early_stopping.best_epoch + 1
        else:
            min_val_idx = self.history["val_total"].index(min(self.history["val_total"]))
            best_epoch = min_val_idx + 1

        components = []
        if "train_mse" in self.history and len(self.history["train_mse"]) > 0:
            components.append(("MSE", "train_mse", "val_mse"))
        if "train_divergence" in self.history and len(self.history["train_divergence"]) > 0:
            components.append(("Divergence", "train_divergence", "val_divergence"))
        if "train_gradient" in self.history and len(self.history["train_gradient"]) > 0:
            components.append(("Gradient", "train_gradient", "val_gradient"))
        if "train_emitter" in self.history and len(self.history["train_emitter"]) > 0:
            components.append(("Emitter", "train_emitter", "val_emitter"))

        n_components = len(components)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, (name, train_key, val_key) in enumerate(components):
            ax = axes[idx]
            ax.plot(epochs, self.history[train_key], color="#0066CC", linewidth=2.0, label="Train", alpha=0.9)
            ax.plot(epochs, self.history[val_key], color="#FF5500", linewidth=2.0, label="Val", alpha=0.9)
            ax.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(f"{name} Loss", fontsize=10)
            ax.set_title(f"{name} Loss", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

            loss_range = max(self.history[train_key]) / (min(self.history[train_key]) + 1e-8)
            if loss_range > 100:
                ax.set_yscale("log")

        for idx in range(n_components, 4):
            axes[idx].axis("off")

        plt.suptitle("Loss Components Over Epochs", fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig

    def plot_metrics_grid(self) -> Figure:
        """Create comprehensive metrics grid plot."""
        num_epochs = len(self.history["val_total"])
        if num_epochs == 0:
            raise ValueError("No history to plot")

        epochs = list(range(1, num_epochs + 1))

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        axes[0].plot(epochs, self.history["val_mse_density"], label="Density", linewidth=2)
        axes[0].plot(epochs, self.history["val_mse_velx"], label="Vel-X", linewidth=2)
        axes[0].plot(epochs, self.history["val_mse_vely"], label="Vel-Y", linewidth=2)
        axes[0].set_title("Per-Channel MSE")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["val_divergence_norm"], color="red", linewidth=2)
        axes[1].axhline(y=0.1, color="green", linestyle="--", label="Target < 0.1")
        axes[1].set_title("Divergence Norm")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("||∇·v||₂")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history["val_kinetic_energy"], color="blue", linewidth=2)
        axes[2].set_title("Kinetic Energy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("KE (trend monitor)")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(epochs, self.history["val_collider_violation"], color="orange", linewidth=2)
        axes[3].axhline(y=0.01, color="green", linestyle="--", label="Target < 0.01")
        axes[3].set_title("Collider Violation")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Density in collider")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(epochs, self.history["val_emitter_accuracy"], color="purple", linewidth=2)
        axes[4].axhline(y=0.1, color="green", linestyle="--", label="Target < 0.1")
        axes[4].set_title("Emitter Density Accuracy")
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Injection error")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        axes[5].plot(epochs, self.history["val_ssim_density"], color="teal", linewidth=2)
        axes[5].axhline(y=0.9, color="green", linestyle="--", label="Target > 0.9")
        axes[5].set_title("SSIM (Density)")
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("SSIM")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        axes[6].plot(epochs, self.history["val_gradient_l1"], color="brown", linewidth=2)
        axes[6].set_title("Gradient L1 (Edge Sharpness)")
        axes[6].set_xlabel("Epoch")
        axes[6].set_ylabel("L1 gradient error")
        axes[6].grid(True, alpha=0.3)

        axes[7].axis("off")

        # Check if log scale needed for MSE
        mse_range = max(self.history["val_mse_density"]) / (min(self.history["val_mse_density"]) + 1e-8)
        if mse_range > 100:
            axes[0].set_yscale("log")

        plt.suptitle("Validation Metrics", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig

    def train(self) -> None:
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Rollout step: K={self.config.rollout_step}")
        if self.config.validation_use_rollout_k:
            print(f"Validation will match training K={self.config.rollout_step}")
        else:
            print("Validation will always use K=1 for consistent metrics")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            cast("FluidNPZSequenceDataset", self.train_loader.dataset).set_epoch(epoch)

            train_losses = self.train_epoch()
            val_losses = self.validate()

            epoch_time = time.time() - epoch_start

            train_metrics = {f"train_{k}": v for k, v in train_losses.items()}
            val_metrics = {f"val_{k}": v for k, v in val_losses.items()}
            mlflow.log_metrics({**train_metrics, **val_metrics, "epoch_time": epoch_time}, step=epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            self.history["train_total"].append(train_losses["total"])
            self.history["train_mse"].append(train_losses.get("mse", 0.0))
            self.history["val_total"].append(val_losses["total"])
            self.history["val_mse"].append(val_losses.get("mse", 0.0))
            self.history["learning_rate"].append(current_lr)

            self.history["val_mse_density"].append(val_losses.get("mse_density", 0.0))
            self.history["val_mse_velx"].append(val_losses.get("mse_velx", 0.0))
            self.history["val_mse_vely"].append(val_losses.get("mse_vely", 0.0))
            self.history["val_divergence_norm"].append(val_losses.get("divergence_norm", 0.0))
            self.history["val_kinetic_energy"].append(val_losses.get("kinetic_energy", 0.0))
            self.history["val_collider_violation"].append(val_losses.get("collider_violation", 0.0))
            self.history["val_emitter_accuracy"].append(val_losses.get("emitter_density_accuracy", 0.0))
            self.history["val_ssim_density"].append(val_losses.get("ssim_density", 0.0))
            self.history["val_gradient_l1"].append(val_losses.get("gradient_l1", 0.0))

            if self.config.physics_loss.enable_divergence:
                self.history["train_divergence"].append(train_losses.get("divergence", 0.0))
                self.history["val_divergence"].append(val_losses.get("divergence", 0.0))

            if self.config.physics_loss.enable_gradient:
                self.history["train_gradient"].append(train_losses.get("gradient", 0.0))
                self.history["val_gradient"].append(val_losses.get("gradient", 0.0))

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_losses['total']:.6f} | "
                f"Val Loss: {val_losses['total']:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            print(
                f"  Train: MSE={train_losses.get('mse', 0):.6f}, "
                f"Div={train_losses.get('divergence', 0):.6f}, "
                f"Grad={train_losses.get('gradient', 0):.6f}, "
                f"Emitter={train_losses.get('emitter', 0):.6f}"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_losses["total"])
                else:
                    self.scheduler.step()

            if self.early_stopping is not None:
                should_stop = self.early_stopping(val_losses["total"], epoch)
                if should_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    self.early_stopping.save_checkpoint(self.model, self.optimizer, self.scaler, self.config, epoch)
                    break

                if self.early_stopping.counter == 0:
                    self.early_stopping.save_checkpoint(self.model, self.optimizer, self.scaler, self.config, epoch)

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

        if self.early_stopping is not None:
            print("Restoring best checkpoint...")
            self.early_stopping.load_best_checkpoint(self.model, self.optimizer, self.scaler, self.device)
        else:
            self.save_checkpoint(self.config.epochs - 1, final=True)

        try:
            fig = self.plot_training_history()
            log_artifact_flat(fig, "training_loss_and_lr.png", dpi=144)
            print("Training history plot saved to MLflow artifacts")
        except Exception as e:
            print(f"Warning: Failed to generate/save training history plot - {e}")

        try:
            fig_metrics = self.plot_metrics_grid()
            log_artifact_flat(fig_metrics, "validation_metrics_grid.png", dpi=90)
            print("Metrics grid plot saved to MLflow artifacts")
        except Exception as e:
            print(f"Warning: Failed to generate/save metrics grid plot - {e}")

        try:
            fig_loss_components = self.plot_loss_components()
            log_artifact_flat(fig_loss_components, "loss_components.png", dpi=90)
            print("Loss components plot saved to MLflow artifacts")
        except Exception as e:
            print(f"Warning: Failed to generate/save loss components plot - {e}")

        print("Training complete!")

    def save_checkpoint(self, epoch: int, final: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
        }

        if self.config.variant is not None:
            checkpoint["variant_metadata"] = self.config.variant.model_dump()

        active_run = mlflow.active_run()
        if active_run:
            checkpoint["mlflow_run_id"] = active_run.info.run_id

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.early_stopping is not None:
            checkpoint["early_stopping_state"] = self.early_stopping.state_dict()

        if final:
            checkpoint_path = self.config.checkpoint_dir / "final_model.pth"
        else:
            checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        mlflow.log_artifact(str(checkpoint_path))

        if not final:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            [f for f in self.config.checkpoint_dir.glob("checkpoint_epoch_*.pth")],
            key=lambda x: x.stat().st_mtime,
        )

        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.early_stopping is not None and "early_stopping_state" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping_state"])

        epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint from epoch {epoch + 1}")

        return epoch
