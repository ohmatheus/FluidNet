import time
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from config.training_config import TrainingConfig


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
        self.config = config
        self.device = device

        # Simple optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

        self.scaler = torch.amp.GradScaler() if config.amp_enabled and device == "cuda" else None

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        """Single epoch training, returns average loss"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in pbar:
            # Data already moved to device in dataset __getitem__
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self) -> float:
        """Validation loop, returns average loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for inputs, targets in pbar:
                # Data already moved to device in dataset __getitem__
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self) -> None:
        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_time = time.time() - epoch_start

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "epoch_time": epoch_time}, step=epoch)

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {epoch_time:.2f}s"
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
        """Remove old checkpoints, keeping only the last N"""
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
        """Load model checkpoint, returns epoch number"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = int(checkpoint.get("epoch", 0))
        print(f"Loaded checkpoint from epoch {epoch + 1}")

        return epoch
