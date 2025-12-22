from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import PROJECT_ROOT_PATH, project_config
from dataset.normalization import load_normalization_scales


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
        seq_indices: list[int] | None = None,
    ) -> None:
        self.npz_dir = npz_dir
        self.normalize = normalize

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

        # Build index mapping and load global normalization scales if needed
        self._index: list[tuple[int, int]] = []  # (seq_idx, t)
        self._norm_scales: dict[str, float] | None = None

        if self.normalize:
            stats_path = PROJECT_ROOT_PATH / project_config.vdb_tools.stats_output_file
            self._norm_scales = load_normalization_scales(stats_path)

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
                assert self._norm_scales is not None, "Normalization scales should be loaded when normalize=True"
                scales = self._norm_scales

                # Apply global percentile-based normalization (no mean centering, just scaling)
                d_t = d_t / scales["S_density"]
                d_tminus = d_tminus / scales["S_density"]
                vx_t = vx_t / scales["S_velx"]
                vz_t = vz_t / scales["S_velz"]

                d_tp1 = d_tp1 / scales["S_density"]
                vx_tp1 = vx_tp1 / scales["S_velx"]
                vz_tp1 = vz_tp1 / scales["S_velz"]

            x = np.stack([d_t, vx_t, vz_t, d_tminus], axis=0)  # (4,H,W)
            y = np.stack([d_tp1, vx_tp1, vz_tp1], axis=0)  # (3,H,W)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        return x_t, y_t
