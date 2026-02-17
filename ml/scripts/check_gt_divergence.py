import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.physics_loss import compute_divergence

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "npz" / "128"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "analysis"


def compute_masked_divergence_rmse(
    velx: np.ndarray, velz: np.ndarray, emitter: np.ndarray, collider: np.ndarray, eps: float = 1e-8
) -> float:
    vx = torch.from_numpy(velx).unsqueeze(0)  # (1, H, W)
    vz = torch.from_numpy(velz).unsqueeze(0)
    div = compute_divergence(vx, vz, mode="central")  # (1, H, W)

    fluid_mask = (torch.from_numpy(emitter) < 0.01) & (torch.from_numpy(collider) < 0.01)
    fluid_mask = fluid_mask.float()

    num_fluid = fluid_mask.sum()
    if num_fluid < eps:
        return 0.0

    masked_div_sq = (div.squeeze(0) ** 2) * fluid_mask
    rmse = torch.sqrt(masked_div_sq.sum() / (num_fluid + eps)).item()
    return rmse


def main() -> None:
    npz_files = sorted(DATA_DIR.glob("seq_*.npz"))
    if not npz_files:
        print(f"No seq_*.npz files found in {DATA_DIR}")
        return

    print(f"Found {len(npz_files)} sequences")

    seq_names = []
    seq_avg_divs = []

    for npz_path in npz_files:
        data = np.load(npz_path)
        velx = data["velx"]  # (T, H, W)
        velz = data["velz"]
        emitter = data.get("emitter", np.zeros_like(velx))
        collider = data.get("collider", np.zeros_like(velx))

        T = velx.shape[0]
        frame_divs = []
        for t in range(T):
            rmse = compute_masked_divergence_rmse(velx[t], velz[t], emitter[t], collider[t])
            frame_divs.append(rmse)

        avg = np.mean(frame_divs)
        seq_names.append(npz_path.stem)
        seq_avg_divs.append(avg)
        print(f"  {npz_path.stem}: {T} frames, avg divergence RMSE = {avg:.6f}")

    avg_divs_arr = np.array(seq_avg_divs)
    print(f"\nOverall: mean={avg_divs_arr.mean():.6f}, min={avg_divs_arr.min():.6f}, max={avg_divs_arr.max():.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(avg_divs_arr, bins=20, color="steelblue", edgecolor="white", alpha=0.9)

    mean_val = avg_divs_arr.mean()
    std_val = avg_divs_arr.std()
    ax.axvline(mean_val, color="#ff6b35", linewidth=2.5, linestyle="-", label=f"Mean: {mean_val:.4f}")
    ax.axvline(mean_val + std_val, color="#ff6b35", linewidth=1.5, linestyle="--", label=f"Std: ±{std_val:.4f}")
    ax.axvline(mean_val - std_val, color="#ff6b35", linewidth=1.5, linestyle="--")

    ax.axvline(0, color="limegreen", linewidth=2, linestyle="-", label="Expected (divergence-free): 0")

    ax.set_xlabel("Avg Divergence RMSE per Sequence", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Ground Truth Divergence Distribution", fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gt_divergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
