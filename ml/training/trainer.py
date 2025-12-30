import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.training_config import TrainingConfig
from training.physics_loss import PhysicsAwareLoss


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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = PhysicsAwareLoss(
            mse_weight=config.physics_loss.mse_weight,
            divergence_weight=config.physics_loss.divergence_weight,
            gradient_weight=config.physics_loss.gradient_weight,
            grid_spacing=config.physics_loss.grid_spacing,
            enable_divergence=config.physics_loss.enable_divergence,
            enable_gradient=config.physics_loss.enable_gradient,
        )

        self.scaler = GradScaler("cuda") if config.amp_enabled and device == "cuda" else None

        self.config.checkpoint_dir = self.config.checkpoint_dir / str(model.model_name)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        loss_accumulators = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.scaler is not None:
                with autocast(device_type=self.device):
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets, inputs)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets, inputs)
                loss.backward()
                self.optimizer.step()

            num_batches += 1

            # Accumulate individual loss components
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
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for inputs, targets in pbar:
                # Move data to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets, inputs)

                num_batches += 1

                # Accumulate individual loss components
                for key, value in loss_dict.items():
                    if key not in loss_accumulators:
                        loss_accumulators[key] = 0.0
                    loss_accumulators[key] += value

                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Average all losses
        avg_losses = {key: value / num_batches for key, value in loss_accumulators.items()}
        return avg_losses

    def train(self) -> None:
        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            train_losses = self.train_epoch()
            val_losses = self.validate()

            epoch_time = time.time() - epoch_start

            # Log all metrics to MLflow
            train_metrics = {f"train_{k}": v for k, v in train_losses.items()}
            val_metrics = {f"val_{k}": v for k, v in val_losses.items()}
            mlflow.log_metrics({**train_metrics, **val_metrics, "epoch_time": epoch_time}, step=epoch)

            # Print summary
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_losses['total']:.6f} | "
                f"Val Loss: {val_losses['total']:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Print detailed breakdown
            print(
                f"  Train: MSE={train_losses.get('mse', 0):.6f}, "
                f"Div={train_losses.get('divergence', 0):.6f}, "
                f"Grad={train_losses.get('gradient', 0):.6f}"
            )

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

        self.save_checkpoint(self.config.epochs - 1, final=True)
        print("Training complete!")

    def save_checkpoint(self, epoch: int, final: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.model_dump(),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint file
        if final:
            checkpoint_path = self.config.checkpoint_dir / "final_model.pth"
        else:
            checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        mlflow.log_artifact(str(checkpoint_path))

        # Clean up old checkpoints
        if not final:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            [f for f in self.config.checkpoint_dir.glob("checkpoint_epoch_*.pth")],
            key=lambda x: x.stat().st_mtime,
        )

        # Remove oldest checkpoints if we exceed the limit
        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint from epoch {epoch + 1}")

        return epoch
