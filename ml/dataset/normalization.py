from pathlib import Path

import yaml


def load_normalization_scales(stats_path: Path) -> dict[str, float]:
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with open(stats_path) as f:
        stats_data = yaml.safe_load(f)

    if "normalization_scales" not in stats_data:
        raise KeyError(f"'normalization_scales' section not found in {stats_path}")

    norm_scales = stats_data["normalization_scales"]

    return {
        "S_density": float(norm_scales["S_density"]),
        "S_velx": float(norm_scales["S_velx"]),
        "S_velz": float(norm_scales["S_velz"]),
    }
