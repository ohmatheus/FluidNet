import time
from pathlib import Path

import numpy as np
import psutil  # type: ignore
import torch
from torch.utils.data import Dataset

from config.config import PROJECT_ROOT_PATH, project_config
from dataset.augmentations import apply_augmentation
from dataset.normalization import load_normalization_scales


def _calculate_fake_count(num_real_sequences: int, fake_empty_pct: int) -> int:
    if fake_empty_pct == 0:
        return 0
    return max(1, int(num_real_sequences * fake_empty_pct / 100))


def _validate_sequence_data(
    path: Path, d: np.ndarray, vx: np.ndarray, vz: np.ndarray, emitter: np.ndarray | None, collider: np.ndarray | None
) -> None:
    if d.ndim != 3 or vx.ndim != 3 or vz.ndim != 3:
        raise ValueError(f"Expected (T,H,W) arrays in {path}")
    if not (d.shape == vx.shape == vz.shape):
        raise ValueError(f"Shape mismatch in {path}: d={d.shape}, vx={vx.shape}, vz={vz.shape}")

    if emitter is not None:
        if emitter.ndim != 3:
            raise ValueError(f"Expected emitter to be (T,H,W) in {path}, got {emitter.shape}")
        if emitter.shape != d.shape:
            raise ValueError(f"Emitter shape mismatch in {path}: emitter={emitter.shape}, density={d.shape}")

    if collider is not None:
        if collider.ndim != 3:
            raise ValueError(f"Expected collider to be (T,H,W) in {path}, got {collider.shape}")
        if collider.shape != d.shape:
            raise ValueError(f"Collider shape mismatch in {path}: collider={collider.shape}, density={d.shape}")

    T = d.shape[0]
    if T < 3:
        raise ValueError(
            f"Animation is less than 3 frames, not enough to form samples. {path}: d={d.shape}, vx={vx.shape}, vz={vz.shape}"
        )


def _load_sequence_metadata(path: Path) -> tuple[int, int, int]:
    with np.load(path) as data:
        d = data["density"]
        vx = data["velx"]
        vz = data["velz"]
        emitter = data["emitter"] if "emitter" in data else None
        collider = data["collider"] if "collider" in data else None

        _validate_sequence_data(path, d, vx, vz, emitter, collider)

        T, H, W = d.shape
        return T, H, W


def _build_real_sequence_indices(seq_paths: list[Path]) -> tuple[list[tuple[int, int]], list[int], int, int]:
    indices: list[tuple[int, int]] = []
    frame_counts: list[int] = []
    h, w = 0, 0

    for si, path in enumerate(seq_paths):
        T, H, W = _load_sequence_metadata(path)

        # Store dimensions from first sequence
        if si == 0:
            h, w = H, W

        frame_counts.append(T)

        # Add valid frame indices [1, T-2]
        for t in range(1, T - 1):
            indices.append((si, t))

    return indices, frame_counts, h, w


def _build_fake_sequence_indices(
    num_fake: int, num_real: int, fake_shape: tuple[int, int, int]
) -> list[tuple[int, int]]:
    indices: list[tuple[int, int]] = []
    fake_t = fake_shape[0]

    for fi in range(num_fake):
        si = num_real + fi
        for t in range(1, fake_t - 1):
            indices.append((si, t))

    return indices


def _apply_normalization(
    d_t: np.ndarray,
    d_tminus: np.ndarray,
    d_tp1: np.ndarray,
    vx_t: np.ndarray,
    vx_tp1: np.ndarray,
    vz_t: np.ndarray,
    vz_tp1: np.ndarray,
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_t = d_t / scales["S_density"]
    d_tminus = d_tminus / scales["S_density"]
    d_tp1 = d_tp1 / scales["S_density"]

    vx_t = vx_t / scales["S_velx"]
    vx_tp1 = vx_tp1 / scales["S_velx"]

    vz_t = vz_t / scales["S_velz"]
    vz_tp1 = vz_tp1 / scales["S_velz"]

    return d_t, d_tminus, d_tp1, vx_t, vx_tp1, vz_t, vz_tp1


def _create_fake_sample(
    t: int, fake_shape: tuple[int, int, int], normalize: bool, norm_scales: dict[str, float] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fake (zero-filled) sample."""
    T, H, W = fake_shape

    # Create zero arrays
    d = np.zeros((T, H, W), dtype=np.float32)
    vx = np.zeros((T, H, W), dtype=np.float32)
    vz = np.zeros((T, H, W), dtype=np.float32)
    emitter = np.zeros((T, H, W), dtype=np.float32)

    # ---------------------------
    # Randomly choose between zeros and ones to avoid collider/density bias (per-sequence randomization)
    collider_value = np.random.choice([0.0, 1.0])
    collider = np.full((T, H, W), collider_value, dtype=np.float32)

    # collider = np.ones((T, H, W), dtype=np.float32)
    # collider = np.zeros((T, H, W), dtype=np.float32)
    # ---------------------------

    # Extract time slices
    d_tminus = d[t - 1]
    d_t = d[t]
    d_tp1 = d[t + 1]

    vx_t = vx[t]
    vx_tp1 = vx[t + 1]

    vz_t = vz[t]
    vz_tp1 = vz[t + 1]

    emitter_t = emitter[t]
    collider_t = collider[t]

    x = np.stack([d_t, vx_t, vz_t, d_tminus, emitter_t, collider_t], axis=0)  # (6,H,W)
    y = np.stack([d_tp1, vx_tp1, vz_tp1], axis=0)  # (3,H,W)

    return torch.from_numpy(x), torch.from_numpy(y)


def _load_sample(
    path: Path, t: int, normalize: bool, norm_scales: dict[str, float] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a real sample from NPZ file."""
    with np.load(path) as data:
        d = data["density"].astype(np.float32, copy=False)
        vx = data["velx"].astype(np.float32, copy=False)
        vz = data["velz"].astype(np.float32, copy=False)

        if "emitter" in data:
            emitter = data["emitter"].astype(np.float32, copy=False)
        else:
            emitter = np.zeros_like(d)

        if "collider" in data:
            collider = data["collider"].astype(np.float32, copy=False)
        else:
            collider = np.zeros_like(d)

        # Extract time slices
        d_tminus = d[t - 1]
        d_t = d[t]
        d_tp1 = d[t + 1]

        vx_t = vx[t]
        vx_tp1 = vx[t + 1]

        vz_t = vz[t]
        vz_tp1 = vz[t + 1]

        emitter_t = emitter[t]
        collider_t = collider[t]

        # Apply normalization if enabled
        if normalize and norm_scales is not None:
            d_t, d_tminus, d_tp1, vx_t, vx_tp1, vz_t, vz_tp1 = _apply_normalization(
                d_t, d_tminus, d_tp1, vx_t, vx_tp1, vz_t, vz_tp1, norm_scales
            )

        x = np.stack([d_t, vx_t, vz_t, d_tminus, emitter_t, collider_t], axis=0)  # (6,H,W)
        y = np.stack([d_tp1, vx_tp1, vz_tp1], axis=0)  # (3,H,W)

    return torch.from_numpy(x), torch.from_numpy(y)


class FluidNPZSequenceDataset(Dataset):
    """
    PyTorch Dataset for fluid sequences saved as seq_*.npz with arrays:
      - density:    (T, H, W)
      - velx:       (T, H, W)
      - velz:       (T, H, W)
      - emitter:    (T, H, W)
      - collider:   (T, H, W)
    """

    def __init__(
        self,
        npz_dir: str | Path,
        normalize: bool = False,
        seq_indices: list[int] | None = None,
        fake_empty_pct: int = 0,
        is_training: bool = False,
        augmentation_config: dict | None = None,
        preload: bool = False,
    ) -> None:
        self.npz_dir = npz_dir
        self.normalize = normalize
        self.fake_empty_pct = fake_empty_pct
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}
        self.enable_augmentation = self.augmentation_config.get("enable_augmentation", False)
        self.flip_probability = self.augmentation_config.get("flip_probability", 0.5)

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

        self.num_real_sequences = len(self.seq_paths)
        self.num_fake_sequences = _calculate_fake_count(self.num_real_sequences, fake_empty_pct)

        # Load global normalization scales
        self._norm_scales: dict[str, float] | None = None
        if self.normalize:
            stats_path = PROJECT_ROOT_PATH / project_config.vdb_tools.stats_output_file
            self._norm_scales = load_normalization_scales(stats_path)

        self._index, frame_counts, h, w = _build_real_sequence_indices(self.seq_paths)

        if not self._index:
            raise RuntimeError("No valid samples found (need T>=3 per sequence)")

        # Add fake sequence indices
        self._fake_shape: tuple[int, int, int] | None
        if self.num_fake_sequences > 0 and frame_counts:
            fake_t = int(np.mean(frame_counts))
            self._fake_shape = (fake_t, h, w)

            fake_indices = _build_fake_sequence_indices(
                self.num_fake_sequences, self.num_real_sequences, self._fake_shape
            )
            self._index.extend(fake_indices)
        else:
            self._fake_shape = None

        self.preload = preload
        self._preloaded_sequences: dict[int, dict[str, np.ndarray]] | None = None

        if self.preload:
            self._preload_sequences()

    def _estimate_memory_usage(self, seq_paths: list[Path]) -> tuple[int, str]:
        total_bytes = 0

        for path in seq_paths:
            with np.load(path) as data:
                for key in ["density", "velx", "velz", "emitter", "collider"]:
                    if key in data:
                        arr = data[key]
                        total_bytes += arr.size * arr.itemsize

        size = float(total_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return total_bytes, f"{size:.2f} {unit}"
            size /= 1024.0

        return total_bytes, f"{size:.2f} TB"

    def _preload_sequences(self) -> None:
        print(f"Preloading {self.num_real_sequences} sequences into memory...")

        mem_bytes, mem_str = self._estimate_memory_usage(self.seq_paths)
        print(f"Estimated memory usage: {mem_str}")

        try:
            available_gb = psutil.virtual_memory().available / (1024**3)
            required_gb = mem_bytes / (1024**3)
            if required_gb > available_gb * 0.8:
                print(f"WARNING: Required memory ({required_gb:.2f} GB) is close to available ({available_gb:.2f} GB)")
        except ImportError:
            pass

        self._preloaded_sequences = {}
        start_time = time.time()

        try:
            for si, path in enumerate(self.seq_paths):
                with np.load(path) as data:
                    seq_data = {
                        "density": data["density"].astype(np.float32, copy=True),
                        "velx": data["velx"].astype(np.float32, copy=True),
                        "velz": data["velz"].astype(np.float32, copy=True),
                        "emitter": data["emitter"].astype(np.float32, copy=True) if "emitter" in data else None,
                        "collider": data["collider"].astype(np.float32, copy=True) if "collider" in data else None,
                    }

                    self._preloaded_sequences[si] = seq_data

        except MemoryError as e:
            print("ERROR: Out of memory during preloading. Falling back to on-disk loading.")
            print("Consider: 1) Setting preload=False, 2) Reducing batch_size, 3) Adding more RAM")
            self._preloaded_sequences = None
            self.preload = False
            raise RuntimeError("Dataset preloading failed due to insufficient memory") from e
        except Exception as e:
            print(f"ERROR: Unexpected error during preloading: {e}")
            self._preloaded_sequences = None
            self.preload = False
            raise

        elapsed = time.time() - start_time
        print(f"Preloaded {self.num_real_sequences} sequences in {elapsed:.2f}s")

    def _load_sample_from_memory(self, si: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._preloaded_sequences is not None, "Preloaded sequences should be available"
        seq_data = self._preloaded_sequences[si]

        d = seq_data["density"]
        vx = seq_data["velx"]
        vz = seq_data["velz"]
        emitter = seq_data["emitter"] if seq_data["emitter"] is not None else np.zeros_like(d)
        collider = seq_data["collider"] if seq_data["collider"] is not None else np.zeros_like(d)

        d_tminus = d[t - 1]
        d_t = d[t]
        d_tp1 = d[t + 1]

        vx_t = vx[t]
        vx_tp1 = vx[t + 1]

        vz_t = vz[t]
        vz_tp1 = vz[t + 1]

        emitter_t = emitter[t]
        collider_t = collider[t]

        if self.normalize and self._norm_scales is not None:
            d_t, d_tminus, d_tp1, vx_t, vx_tp1, vz_t, vz_tp1 = _apply_normalization(
                d_t, d_tminus, d_tp1, vx_t, vx_tp1, vz_t, vz_tp1, self._norm_scales
            )

        x = np.stack([d_t, vx_t, vz_t, d_tminus, emitter_t, collider_t], axis=0)
        y = np.stack([d_tp1, vx_tp1, vz_tp1], axis=0)

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        si, t = self._index[idx]

        if si >= self.num_real_sequences:
            assert self._fake_shape is not None, "Fake shape should be set when fake sequences are enabled"
            x, y = _create_fake_sample(t, self._fake_shape, self.normalize, self._norm_scales)
        else:
            if self.preload and self._preloaded_sequences is not None:
                x, y = self._load_sample_from_memory(si, t)
            else:
                path = self.seq_paths[si]
                x, y = _load_sample(path, t, self.normalize, self._norm_scales)

        if self.is_training and self.enable_augmentation:
            x, y = apply_augmentation(x, y, self.flip_probability)

        return x, y
