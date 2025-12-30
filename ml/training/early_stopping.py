from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, checkpoint_dir: Path) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = -1

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def save_checkpoint(
        self, model: nn.Module, optimizer: Any, scaler: GradScaler | None, config: Any, epoch: int
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.model_dump(),
            "val_loss": self.best_loss,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, path)
        print(f"Saved best checkpoint (val_loss={self.best_loss:.6f}) at epoch {epoch + 1}")

    def load_best_checkpoint(self, model: nn.Module, optimizer: Any, scaler: GradScaler | None, device: str) -> None:
        path = self.checkpoint_dir / "best_model.pth"
        if not path.exists():
            print("Warning: No best checkpoint found, skipping restoration")
            return

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Restored best checkpoint from epoch {checkpoint['epoch'] + 1} (val_loss={checkpoint['val_loss']:.6f})")

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counter = state["counter"]
        self.best_loss = state["best_loss"]
        self.best_epoch = state["best_epoch"]
