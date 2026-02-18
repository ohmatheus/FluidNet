import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from config.config import PROJECT_ROOT_PATH
from config.training_config import TrainingConfig, project_config
from dataset.normalization import load_normalization_scales
from models.unet import UNet, UNetConfig
from training.test_evaluation import (
    ROLLOUT_STEPS,
    _get_starting_frames,
    _plot_rollout_degradation,
    _run_single_rollout,
)

ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT.parent / "data" / "npz" / "128"
OUTPUT_DIR = ML_ROOT.parent / "data" / "analysis"
BASE_CONFIG_PATH = ML_ROOT / "config" / "base_config.yaml"


def load_from_checkpoint(checkpoint_dir: Path, device: str) -> tuple[TrainingConfig, torch.nn.Module]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best_model.pth in {checkpoint_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainingConfig(**checkpoint["config"])
    config.device = device

    model = UNet(
        cfg=UNetConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            depth=config.depth,
            norm=config.norm,
            act=config.act,
            group_norm_groups=config.group_norm_groups,
            dropout=config.dropout,
            upsample=config.upsample,
            downsample=config.downsample,
            padding_mode=config.padding_mode,
            use_residual=config.use_residual,
            bottleneck_blocks=config.bottleneck_blocks,
            output_activation=config.output_activation,
        )
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded: {checkpoint_path}")

    return config, model


def run_rollout_to_disk(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: str,
    output_name: str,
) -> None:
    test_data_dir = DATA_DIR / "test"
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test split directory not found: {test_data_dir}")

    test_paths = sorted([p for p in test_data_dir.iterdir() if p.name.startswith("seq_") and p.name.endswith(".npz")])

    norm_scales = None
    if config.normalize:
        stats_path = PROJECT_ROOT_PATH / project_config.vdb_tools.stats_output_file
        norm_scales = load_normalization_scales(stats_path)

    print(f"Sequences: {len(test_paths)} | Steps: {ROLLOUT_STEPS}")

    all_rollouts: list[list[dict[str, float]]] = [[] for _ in range(ROLLOUT_STEPS)]
    total_rollouts = 0

    model.eval()
    with torch.no_grad():
        for seq_path in test_paths:
            with np.load(seq_path) as npz:
                data = {
                    "density": npz["density"].astype(np.float32),
                    "velx": npz["velx"].astype(np.float32),
                    "velz": npz["velz"].astype(np.float32),
                    "emitter": npz.get("emitter", np.zeros_like(npz["density"])).astype(np.float32),
                    "collider": npz.get("collider", np.zeros_like(npz["density"])).astype(np.float32),
                }

            T = data["density"].shape[0]
            starting_frames = _get_starting_frames(T)

            for t_start in starting_frames:
                step_metrics = _run_single_rollout(
                    model, data, t_start, norm_scales, device, config.amp_enabled, config
                )
                for k, m in enumerate(step_metrics):
                    all_rollouts[k].append(m)
                total_rollouts += 1

            print(f"  {seq_path.stem}: {len(starting_frames)} rollouts")

    avg_per_step: list[dict[str, float]] = []
    metric_keys = [
        "mse_density",
        "mse_velx",
        "mse_vely",
        "ssim_density",
        "divergence_norm",
        "divergence_norm_gt",
        "gradient_l1",
    ]
    for k in range(ROLLOUT_STEPS):
        if not all_rollouts[k]:
            avg_per_step.append({key: 0.0 for key in metric_keys})
            continue
        avg_per_step.append({key: sum(r[key] for r in all_rollouts[k]) / len(all_rollouts[k]) for key in metric_keys})

    print(f"\nTotal rollouts: {total_rollouts}")

    fig = _plot_rollout_degradation(avg_per_step)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"rollout_{output_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollout evaluation on an existing checkpoint")
    parser.add_argument("checkpoint_dir", type=str, help="Path to checkpoint folder containing best_model.pth")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output-name", type=str, default=None, help="Output filename suffix (default: folder name)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    config, model = load_from_checkpoint(checkpoint_dir, device)

    output_name = args.output_name or checkpoint_dir.name
    run_rollout_to_disk(model, config, device, output_name)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
