import importlib.util
from pathlib import Path

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


class VDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", frozen=True)

    BLENDER_PATH: Path = Path("")


vdb_config = VDBSettings()

__all__ = [
    "PROJECT_ROOT_PATH",
    "project_config",
    "vdb_config",
]
