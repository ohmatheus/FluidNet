import importlib.util
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent
_SHARED_CONFIG_PATH = _PROJECT_ROOT / "shared" / "config.py"

spec = importlib.util.spec_from_file_location("shared_config", _SHARED_CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load shared config from {_SHARED_CONFIG_PATH}")
shared_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_config)

PROJECT_ROOT_PATH = shared_config.PROJECT_ROOT_PATH
project_config = shared_config.project_config


class SplitConfig(BaseModel):
    enabled: bool = True
    ratios: list[float] = [0.70, 0.15, 0.15]
    names: list[str] = ["train", "val", "test"]
    seed: int = 42


class DistributionConfig(BaseModel):
    no_emitter_pct: float = 0.05
    no_collider_pct: float = 0.50
    collider_mode_simple_threshold: float = 0.20
    collider_mode_medium_threshold: float = 0.80


class ScaleConfig(BaseModel):
    min: float
    max: float
    y_scale: float = 0.1


class EmitterScaleConfig(ScaleConfig):
    max_simple_mode: float = 0.2


class PositionConfig(BaseModel):
    x_range: tuple[float, float]
    z_range: tuple[float, float]


class ColliderPositionConfig(BaseModel):
    z_range: tuple[float, float]


class LargeEmitterConfig(BaseModel):
    threshold: float = 0.12
    x_range: tuple[float, float] = (-0.6, 0.6)
    z_position: float = -0.75


class EmitterConfig(BaseModel):
    count_range: tuple[int, int]
    scale: EmitterScaleConfig
    position: PositionConfig
    large_emitter: LargeEmitterConfig = LargeEmitterConfig()


class ColliderModeConfig(BaseModel):
    count_range: tuple[int, int]


class ColliderScaledModeConfig(ColliderModeConfig):
    scale: ScaleConfig


class ColliderConfig(BaseModel):
    simple_mode: ColliderScaledModeConfig
    medium_mode: ColliderModeConfig
    complex_mode: ColliderScaledModeConfig
    position: ColliderPositionConfig


class VorticityConfig(BaseModel):
    range: float | list[float]
    step: float = 0.1


class DomainConfig(BaseModel):
    y_scale: float = 0.05
    vorticity: VorticityConfig = VorticityConfig(range=0.05)
    beta: float = 0.0


class AnimationConfig(BaseModel):
    max_displacement: float = 1e-5


class SimulationGenerationConfig(BaseModel):
    splits: SplitConfig = SplitConfig()
    distribution: DistributionConfig = DistributionConfig()
    emitters: EmitterConfig
    colliders: ColliderConfig
    domain: DomainConfig = DomainConfig()
    animation: AnimationConfig = AnimationConfig()


def get_vorticity_levels(v: VorticityConfig) -> list[float]:
    if isinstance(v.range, (int, float)):
        return [float(v.range)]
    import numpy as np
    return np.arange(v.range[0], v.range[1] + v.step / 2, v.step).tolist()


def load_simulation_config() -> SimulationGenerationConfig:
    config_path = Path(__file__).parent / "config" / "simulation_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return SimulationGenerationConfig(**config_data["simulation_generation"])


simulation_config = load_simulation_config()


class VDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    BLENDER_PATH: Path = Path("")


vdb_config = VDBSettings()

__all__ = [
    "PROJECT_ROOT_PATH",
    "project_config",
    "vdb_config",
    "simulation_config",
    "VorticityConfig",
    "get_vorticity_levels",
]
