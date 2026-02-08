import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.physics_loss import compute_divergence


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "npz" / "128"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "analysis"


def compute_masked_divergence_rmse(velx, velz, emitter, collider, eps=1e-8):
    vx = torch.from_numpy(velx).unsqueeze(0)  # (1, H, W)
    vz = torch.from_numpy(velz).unsqueeze(0)
    div = compute_divergence(vx, vz)  # (1, H, W)

    fluid_mask = (torch.from_numpy(emitter) < 0.01) & (torch.from_numpy(collider) < 0.01)
    fluid_mask = fluid_mask.float()

    num_fluid = fluid_mask.sum()
    if num_fluid < eps:
        return 0.0

    masked_div_sq = (div.squeeze(0) ** 2) * fluid_mask
    rmse = torch.sqrt(masked_div_sq.sum() / (num_fluid + eps)).item()
    return rmse


def main():
    npz_files = sorted(DATA_DIR.glob("seq_*.npz"))
    if not npz_files:
        print(f"No seq_*.npz files found in {DATA_DIR}")
        return

    print(f"Found {len(npz_files)} sequences")

    seq_names = []
    seq_avg_divs = []

    for npz_path in npz_files:
        data = np.load(npz_path)
        velx = data["velx"]      # (T, H, W)
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

    seq_avg_divs = np.array(seq_avg_divs)
    print(f"\nOverall: mean={seq_avg_divs.mean():.6f}, min={seq_avg_divs.min():.6f}, max={seq_avg_divs.max():.6f}")

    fig, ax = plt.subplots(figsize=(max(8, len(seq_names) * 0.4), 5))
    ax.bar(range(len(seq_names)), seq_avg_divs, color="steelblue")
    ax.axhline(y=0.1, color="red", linestyle="--", label="threshold (0.1)")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Avg Divergence RMSE (fluid-only)")
    ax.set_title("Ground Truth Divergence per Sequence")
    ax.set_xticks(range(len(seq_names)))
    ax.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=7)
    ax.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gt_divergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
