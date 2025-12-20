from pathlib import Path

from pydantic import BaseModel

from config.config import PROJECT_ROOT_PATH, project_config


class TrainingConfig(BaseModel):
    # Basic training settings
    batch_size: int = 4
    learning_rate: float = 0.001
    epochs: int = 10
    device: str | None = "cuda"
    amp_enabled: bool = True
    num_workers: int = 4

    # Dataset settings
    npz_dir: Path = Path(PROJECT_ROOT_PATH / "vdb-tools/numpy_output/")
    normalize: bool = True
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train, val, test - to change
    split_seed: int = 42

    # Model architecture
    in_channels: int = project_config.simulation.input_channels
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 2

    # MLFlow settings
    mlflow_tracking_uri: str = "./mlruns"  # mlflow server
    mlflow_experiment_name: str = "fluid_baseline_v1"

    # Checkpoint settings
    checkpoint_dir: Path = Path(PROJECT_ROOT_PATH / "data/checkpoints")
    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3
