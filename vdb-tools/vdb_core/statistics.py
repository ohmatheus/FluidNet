from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

FieldName = Literal["density", "velx", "velz"]


@dataclass
class FieldStats:
    min: float
    max: float
    mean: float
    median: float
    percentiles: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "median": float(self.median),
            "percentiles": {int(k): float(v) for k, v in self.percentiles.items()},
        }


@dataclass
class SequenceStats:
    sequence_name: str
    density: FieldStats
    velx: FieldStats
    velz: FieldStats

    def to_dict(self) -> dict:
        return {
            "sequence_name": self.sequence_name,
            "density": self.density.to_dict(),
            "velx": self.velx.to_dict(),
            "velz": self.velz.to_dict(),
        }


@dataclass
class GlobalStats:
    num_sequences: int
    density: FieldStats
    velx: FieldStats
    velz: FieldStats

    def to_dict(self) -> dict:
        return {
            "num_sequences": self.num_sequences,
            "density": self.density.to_dict(),
            "velx": self.velx.to_dict(),
            "velz": self.velz.to_dict(),
        }


@dataclass
class NormalizationScales:
    S_density: float
    S_velx: float
    S_velz: float
    percentile_used: int

    def to_dict(self) -> dict:
        return {
            "S_density": float(self.S_density),
            "S_velx": float(self.S_velx),
            "S_velz": float(self.S_velz),
        }


def compute_field_stats(array: np.ndarray, percentiles: list[int], use_absolute: bool = False) -> FieldStats:
    # Flatten to 1D for statistics
    flat = array.ravel()

    # For velocity fields, percentiles should be on absolute values
    # but min/max/mean/median remain on raw values
    abs_flat = np.abs(flat) if use_absolute else flat

    stats = FieldStats(
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        mean=float(np.mean(flat)),
        median=float(np.median(flat)),
        percentiles={p: float(np.percentile(abs_flat, p)) for p in percentiles},
    )

    return stats


def compute_sequence_stats(
    sequence_name: str,
    density: np.ndarray,
    velx: np.ndarray,
    velz: np.ndarray,
    percentiles: list[int],
) -> SequenceStats:
    return SequenceStats(
        sequence_name=sequence_name,
        density=compute_field_stats(density, percentiles, use_absolute=False),
        velx=compute_field_stats(velx, percentiles, use_absolute=True),
        velz=compute_field_stats(velz, percentiles, use_absolute=True),
    )


def aggregate_global_stats(sequence_stats_list: list[SequenceStats], percentiles: list[int]) -> GlobalStats:
    if not sequence_stats_list:
        raise ValueError("Cannot compute global stats from empty sequence list")

    # Extract per-sequence values for each field
    def aggregate_field(field_name: FieldName) -> FieldStats:
        # Collect all per-sequence stats for this field
        field_stats = [getattr(seq_stat, field_name) for seq_stat in sequence_stats_list]

        # Aggregate
        global_min = min(fs.min for fs in field_stats)
        global_max = max(fs.max for fs in field_stats)
        global_mean = float(np.mean([fs.mean for fs in field_stats]))
        global_median = float(np.median([fs.median for fs in field_stats]))

        # For percentiles, take percentile of per-sequence percentile values
        global_percentiles = {}
        for p in percentiles:
            per_seq_p_values = [fs.percentiles[p] for fs in field_stats]
            global_percentiles[p] = float(np.percentile(per_seq_p_values, p))

        return FieldStats(
            min=float(global_min),
            max=float(global_max),
            mean=global_mean,
            median=global_median,
            percentiles=global_percentiles,
        )

    return GlobalStats(
        num_sequences=len(sequence_stats_list),
        density=aggregate_field("density"),
        velx=aggregate_field("velx"),
        velz=aggregate_field("velz"),
    )


def compute_normalization_scales(global_stats: GlobalStats, normalization_percentile: int) -> NormalizationScales:
    if normalization_percentile not in global_stats.density.percentiles:
        raise ValueError(
            f"Percentile {normalization_percentile} not found in global statistics. "
            f"Available percentiles: {list(global_stats.density.percentiles.keys())}"
        )

    return NormalizationScales(
        S_density=global_stats.density.percentiles[normalization_percentile],
        S_velx=global_stats.velx.percentiles[normalization_percentile],
        S_velz=global_stats.velz.percentiles[normalization_percentile],
        percentile_used=normalization_percentile,
    )


def save_stats_to_yaml(
    output_path: Path,
    sequence_stats: list[SequenceStats],
    global_stats: GlobalStats,
    normalization_scales: NormalizationScales,
) -> None:
    output_data = {
        "metadata": {
            "description": "Field statistics for VDB-to-numpy conversion pipeline",
            "fields": ["density", "velx", "velz"],
            "normalization_percentile": normalization_scales.percentile_used,
        },
        "normalization_scales": normalization_scales.to_dict(),
        "global_stats": global_stats.to_dict(),
        "per_sequence_stats": [seq_stat.to_dict() for seq_stat in sequence_stats],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nStatistics saved to: {output_path}")
