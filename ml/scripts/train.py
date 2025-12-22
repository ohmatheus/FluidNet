import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from config.training_config import TrainingConfig
from dataset.npz_sequence import FluidNPZSequenceDataset
from models.small_unet import SmallUNet
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

    n_train = max(1, int(split_ratios[0] * n_seq))
    n_val = max(1, int(split_ratios[1] * n_seq))

    if n_train + n_val >= n_seq:
        n_train = n_seq - 2  # leave room for val and test in case of few seq
        n_val = 1

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

    train_idx, val_idx, test_idx = make_splits(config.npz_dir, config.split_ratios, config.split_seed)

    print(f"Dataset splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = FluidNPZSequenceDataset(
        npz_dir=config.npz_dir, normalize=config.normalize, seq_indices=train_idx, fake_empty_pct=5
    )
    val_ds = FluidNPZSequenceDataset(
        npz_dir=config.npz_dir, normalize=config.normalize, seq_indices=val_idx, fake_empty_pct=5
    )

    # to compare models later
    # test_ds = FluidNPZSequenceDataset(
    #    npz_dir=config.npz_dir, normalize=config.normalize, seq_indices=test_idx, fake_empty_pct=5
    # )

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = SmallUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        depth=config.depth,
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name="dummy"):
        mlflow.log_params(
            {
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "model_params": total_params,
                "base_channels": config.base_channels,
                "depth": config.depth,
                "normalize": config.normalize,
                "split_seed": config.split_seed,
                "amp_enabled": config.amp_enabled,
                "device": config.device,
            }
        )

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=config.device
        )

        trainer.train()

    print("Training complete!")


if __name__ == "__main__":
    # Must be set before any CUDA operations when using DataLoader with num_workers > 0
    mp.set_start_method("spawn", force=True)
    main()
