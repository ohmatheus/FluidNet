from pathlib import Path

from pydantic import BaseModel, field_validator

from config.config import PROJECT_ROOT_PATH, project_config
from models.unet import ActType, NormType, UpsampleType


class PhysicsLossConfig(BaseModel):
    mse_weight: float = 1.0
    divergence_weight: float = 0.005
    gradient_weight: float = 0.002
    emitter_weight: float = 0.15

    enable_divergence: bool = True
    enable_gradient: bool = False
    enable_emitter: bool = True

    grid_spacing: float = 2.0 / project_config.simulation.grid_resolution


class AugmentationConfig(BaseModel):
    enable_augmentation: bool = True
    flip_probability: float = 0.5
    flip_axis: str = "x"


class TrainingConfig(BaseModel):
    batch_size: int = 4
    learning_rate: float = 0.0007
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
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0

    # Dataset settings
    npz_dir: Path = Path(PROJECT_ROOT_PATH / project_config.vdb_tools.npz_output_directory)
    normalize: bool = True
    split_ratios: tuple[float, float, float] = (0.80, 0.20, 0)  # train, val, test - to change
    split_seed: int = 42
    augmentation: AugmentationConfig = AugmentationConfig()
    preload_dataset: bool = True

    # Model architecture
    in_channels: int = project_config.simulation.input_channels
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 3
    norm: NormType = "instance"  # "none", "batch", "instance", "group"
    act: ActType = "gelu"  # "relu", "leaky_relu", "gelu", "silu"
    group_norm_groups: int = 8
    dropout: float = 0.0
    upsample: UpsampleType = "nearest"  # "nearest", "bilinear", "transpose"
    use_residual: bool = True
    bottleneck_blocks: int = 1
    output_activation: bool = True

    # MLFlow settings
    mlflow_tracking_uri: str = "./mlruns"  # mlflow server
    mlflow_experiment_name: str = "fluid_baseline_v1"

    # Checkpoint settings
    checkpoint_dir: Path = Path(PROJECT_ROOT_PATH / project_config.models.pytorch_folder)
    save_every_n_epochs: int = 2
    keep_last_n_checkpoints: int = 10

    # Physics-aware loss configuration
    physics_loss: PhysicsLossConfig = PhysicsLossConfig()

    # Multi-step rollout training
    rollout_schedule: dict[int, int] = {
        0: 1,  # k1
        4: 2,  # k2
        8: 3,  # k3
        12: 4,  # k4
    }
    rollout_steps: int = 1
    rollout_weight_decay: float = 1.50  # not used if rollout_final_step_only is True
    rollout_gradient_truncation: bool = False
    rollout_reset_lr_on_k_change: bool = True
    validation_use_rollout_k: bool = True
    rollout_final_step_only: bool = False

    @field_validator("rollout_schedule")
    @classmethod
    def validate_rollout_schedule(cls, v: dict[int, int]) -> dict[int, int]:
        if not v:
            raise ValueError("rollout_schedule cannot be empty")
        if 0 not in v:
            raise ValueError("rollout_schedule must contain epoch 0")
        for epoch, K in v.items():
            if epoch < 0:
                raise ValueError(f"Epoch {epoch} must be >= 0")
            if K < 1:
                raise ValueError(f"K={K} at epoch {epoch} must be >= 1")
        return v

    @field_validator("rollout_steps")
    @classmethod
    def validate_rollout_steps(cls, v: int) -> int:
        # add check is <= rollout_schedule lenght
        if v < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {v}")
        return v
