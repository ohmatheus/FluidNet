from pathlib import Path

from pydantic import BaseModel

from config.config import PROJECT_ROOT_PATH, project_config
from models.small_unet_full import ActType, NormType, UpsampleType


class PhysicsLossConfig(BaseModel):
    mse_weight: float = 1.0
    divergence_weight: float = 0.01
    gradient_weight: float = 0.001
    emitter_weight: float = 0.1

    enable_divergence: bool = True
    enable_gradient: bool = False
    enable_emitter: bool = True

    grid_spacing: float = 2.0 / project_config.simulation.grid_resolution


class TrainingConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 0.0004
    epochs: int = 200
    device: str | None = "cuda"
    amp_enabled: bool = False
    num_workers: int = 4

    gradient_clip_norm: float = 1.0
    gradient_clip_enabled: bool = True

    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "plateau"
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    lr_scheduler_step_size: int = 30
    lr_scheduler_t_max: int = 100

    use_early_stopping: bool = True
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0

    # Dataset settings
    npz_dir: Path = Path(PROJECT_ROOT_PATH / project_config.vdb_tools.npz_output_directory)
    normalize: bool = True
    split_ratios: tuple[float, float, float] = (0.80, 0.20, 0)  # train, val, test - to change
    split_seed: int = 42
    fake_empty_pct: int = 5

    # Model architecture
    in_channels: int = project_config.simulation.input_channels
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 3
    norm: NormType = "instance"  # "none", "batch", "instance", "group"
    act: ActType = "gelu"  # "relu", "leaky_relu", "gelu", "silu"
    group_norm_groups: int = 8
    dropout: float = 0.0
    upsample: UpsampleType = "bilinear"  # "nearest", "bilinear", "transpose"
    use_residual: bool = True
    bottleneck_blocks: int = 1
    output_activation: bool = True

    # MLFlow settings
    mlflow_tracking_uri: str = "./mlruns"  # mlflow server
    mlflow_experiment_name: str = "fluid_baseline_v1"

    # Checkpoint settings
    checkpoint_dir: Path = Path(PROJECT_ROOT_PATH / project_config.models.pytorch_folder)
    save_every_n_epochs: int = 3
    keep_last_n_checkpoints: int = 20

    # Physics-aware loss configuration
    physics_loss: PhysicsLossConfig = PhysicsLossConfig()
