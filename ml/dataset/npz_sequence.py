from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FluidNPZSequenceDataset(Dataset):
    """
    PyTorch Dataset for fluid sequences saved as seq_*.npz with arrays:
      - density: (T, H, W)
      - velx:    (T, H, W)
      - velz:    (T, H, W)

    Each sample corresponds to a time index t in [1, T-2] and yields:
      input:  (4, H, W) = [density_t, velx_t, velz_t, density_{t-1}]
      target: (3, H, W) = [density_{t+1}, velx_{t+1}, velz_{t+1}]
    """

    def __init__(
        self,
        npz_dir: str | Path,
        normalize: bool = False,
        device: str = "cpu",
        seq_indices: list[int] | None = None,
    ) -> None:
        self.npz_dir = npz_dir
        self.normalize = normalize
        self.device = torch.device(device) if device is not None else None

        npz_dir_path = Path(npz_dir)
        all_seq_paths: list[Path] = sorted(
            [f for f in npz_dir_path.iterdir() if f.name.startswith("seq_") and f.name.endswith(".npz")]
        )
        if not all_seq_paths:
            raise FileNotFoundError(f"No seq_*.npz files found in {npz_dir}")

        # Filter to specified sequences if provided (splits)
        if seq_indices is not None:
            self.seq_paths = [all_seq_paths[i] for i in seq_indices]
        else:
            self.seq_paths = all_seq_paths

        # Build index mapping and optionally compute per-sequence stats
        self._index: list[tuple[int, int]] = []  # (seq_idx, t)
        self._stats: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(self.seq_paths)

        for si, path in enumerate(self.seq_paths):
            with np.load(path) as data:
                d = data["density"]  # (T,H,W)
                vx = data["velx"]
                vz = data["velz"]

                if d.ndim != 3 or vx.ndim != 3 or vz.ndim != 3:
                    raise ValueError(f"Expected (T,H,W) arrays in {path}")
                if not (d.shape == vx.shape == vz.shape):
                    raise ValueError(f"Shape mismatch in {path}: d={d.shape}, vx={vx.shape}, vz={vz.shape}")

                T = d.shape[0]
                if T < 3:
                    raise ValueError(
                        f"Animation is less than 3 frames, not enough to form samples. {path}: d={d.shape}, vx={vx.shape}, vz={vz.shape}"
                    )

                # Indices t in [1, T-2]
                for t in range(1, T - 1):
                    if t <= T - 2:
                        self._index.append((si, t))

                if self.normalize:
                    # Compute per-sequence channel means/stds over (T,H,W)
                    # channels: [density, velx, velz]
                    c_means = np.array(
                        [
                            d.mean(dtype=np.float32),
                            vx.mean(dtype=np.float32),
                            vz.mean(dtype=np.float32),
                        ],
                        dtype=np.float32,
                    )
                    c_stds = np.array(
                        [
                            d.std(dtype=np.float32),
                            vx.std(dtype=np.float32),
                            vz.std(dtype=np.float32),
                        ],
                        dtype=np.float32,
                    )
                    # Avoid divide by zero
                    c_stds = np.where(c_stds == 0, 1.0, c_stds)
                    self._stats[si] = (c_means, c_stds)

        if not self._index:
            raise RuntimeError("No valid samples found (need T>=3 per sequence)")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        si, t = self._index[idx]
        path = self.seq_paths[si]

        with np.load(path) as data:
            d = data["density"].astype(np.float32, copy=False)
            vx = data["velx"].astype(np.float32, copy=False)
            vz = data["velz"].astype(np.float32, copy=False)

            d_tminus = d[t - 1]
            d_t = d[t]
            vx_t = vx[t]
            vz_t = vz[t]

            d_tp1 = d[t + 1]
            vx_tp1 = vx[t + 1]
            vz_tp1 = vz[t + 1]

            if self.normalize:
                stats = self._stats[si]
                assert stats is not None, f"Stats should be computed for sequence {si} when normalize=True"
                means, stds = stats

                # Apply per-channel normalization - todo
                d_t = (d_t - means[0]) / stds[0]
                d_tminus = (d_tminus - means[0]) / stds[0]
                vx_t = (vx_t - means[1]) / stds[1]
                vz_t = (vz_t - means[2]) / stds[2]

                d_tp1 = (d_tp1 - means[0]) / stds[0]
                vx_tp1 = (vx_tp1 - means[1]) / stds[1]
                vz_tp1 = (vz_tp1 - means[2]) / stds[2]

            x = np.stack([d_t, vx_t, vz_t, d_tminus], axis=0)  # (4,H,W)
            y = np.stack([d_tp1, vx_tp1, vz_tp1], axis=0)  # (3,H,W)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.device is not None:
            x_t = x_t.to(self.device, non_blocking=False)
            y_t = y_t.to(self.device, non_blocking=False)

        return x_t, y_t
