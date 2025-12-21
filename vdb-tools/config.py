from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT_PATH = Path(__file__).parent.parent


class VDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    BLENDER_PATH: Path = Path("")


# -------------------
class ProjectVDBConfig(BaseModel):
    blender_cache_directory: Path
    npz_output_directory: Path


class SimulationConfig(BaseModel):
    grid_resolution: int
    input_channels: int


class ProjectConfig(BaseModel):
    simulation: SimulationConfig
    vdb_tools: ProjectVDBConfig


def load_project_config() -> ProjectConfig:
    config_path = PROJECT_ROOT_PATH / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    return ProjectConfig(**config_data)


# -------------------

vdb_config = VDBSettings()
project_config = load_project_config()
