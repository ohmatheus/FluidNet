from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ML_ROOT_PATH = Path(__file__).parent.parent
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)


# === project .yaml config ===

class SimulationConfig(BaseModel):
    grid_resolution: list[int]


class ProjectConfig(BaseModel):
    simulation: SimulationConfig


def load_project_config() -> ProjectConfig:
    """Load and parse the project configuration from config.yaml."""
    config_path = PROJECT_ROOT_PATH / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    return ProjectConfig(**config_data)


ml_config = MLSettings()
project_config = load_project_config()
