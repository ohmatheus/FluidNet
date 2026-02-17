from pathlib import Path

from pydantic import BaseModel, field_validator

from config.config import PROJECT_ROOT_PATH, project_config
from models.unet import ActType, DownsampleType, NormType, OutputActivationType, PaddingType, UpsampleType
from training.physics_loss import StencilMode


class PhysicsLossConfig(BaseModel):
    mse_weight: float = 1.0
    divergence_weight: float = 0.001
    gradient_weight: float = 0.002
    emitter_weight: float = 0.05

    enable_divergence: bool = True
    enable_gradient: bool = False
    enable_emitter: bool = False
    stencil_mode: StencilMode = "forward"

    grid_spacing: float = 2.0 / project_config.simulation.grid_resolution


class AugmentationConfig(BaseModel):
    enable_augmentation: bool = True
    flip_probability: float = 0.5
    flip_axis: str = "x"


class VariantMetadata(BaseModel):
    variant_name: str
    model_architecture_name: str
    full_model_name: str
    parent_variant: str | None = None
    warmstart_checkpoint: Path | None = None
    variant_yaml_path: Path
    relative_dir: Path


class TrainingConfig(BaseModel):
    batch_size: int = 4
    learning_rate: float = 0.0007
    epochs: int = 200
    device: str | None = "cuda"
    amp_enabled: bool = True
    num_workers: int = 4

    # Dataset settings
    npz_dir: Path = Path(PROJECT_ROOT_PATH / project_config.vdb_tools.npz_output_directory)
    normalize: bool = True
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
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
    upsample: UpsampleType = "bilinear"  # "nearest", "bilinear", "transpose"
    downsample: DownsampleType = "stride"  # "stride", "avgpool", "maxpool"
    padding_mode: PaddingType = "replicate"  # "zeros", "reflect", "replicate", "circular"
    use_residual: bool = True
    bottleneck_blocks: int = 1
    output_activation: OutputActivationType = "linear_clamp"

    # MLFlow settings
    mlflow_tracking_uri: str = "./mlruns"  # mlflow server
    mlflow_experiment_name: str = "fluid_baseline_v1"

    # Checkpoint settings
    checkpoint_dir: Path = Path(PROJECT_ROOT_PATH / project_config.models.pytorch_folder)
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 20

    # Training config
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

    physics_loss: PhysicsLossConfig = PhysicsLossConfig()

    variant: VariantMetadata | None = None

    # Multi-step rollout training
    rollout_step: int = 0
    rollout_weight_decay: float = 1.10  # not used if rollout_final_step_only is True
    #    ∂total_loss/∂θ =
    #    w₀/sum × ∂loss₀/∂θ                     [direct from step 0]
    #  + w₁/sum × ∂loss₁/∂θ                     [direct from step 1]
    #  + w₁/sum × ∂loss₁/∂pred₀ × ∂pred₀/∂θ     [indirect: step 1 through step 0]
    #  + w₂/sum × ∂loss₂/∂θ                     [direct from step 2]
    #  + w₂/sum × ∂loss₂/∂pred₁ × ∂pred₁/∂θ     [indirect: step 2 through step 1]
    #  + w₂/sum × ∂loss₂/∂pred₀ × ∂pred₀/∂θ     [indirect: step 2 through step 0]

    rollout_gradient_truncation: bool = False
    validation_use_rollout_k: bool = True
    rollout_final_step_only: bool = False  # True use .detach() and makes backward pass indenpendent for each K - Faster, less memory, but less quality learning

    @field_validator("rollout_step")
    @classmethod
    def validate_rollout_step(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"rollout_step must be >= 0, got {v}")
        return v

    @property
    def checkpoint_dir_variant(self) -> Path:
        if self.variant:
            return self.checkpoint_dir / self.variant.relative_dir / self.variant.full_model_name
        return self.checkpoint_dir / "Unet"  # Backward compatibility

    @property
    def inference_output_dir_variant(self) -> Path:
        if self.variant:
            return (
                Path(PROJECT_ROOT_PATH)
                / "data"
                / "simple-infer-output"
                / self.variant.relative_dir
                / self.variant.full_model_name
            )
        return Path(PROJECT_ROOT_PATH) / "data" / "simple-infer-output"

    @property
    def onnx_export_path_variant(self) -> Path:
        if self.variant:
            onnx_dir = Path(PROJECT_ROOT_PATH) / "data" / "onnx-export" / self.variant.relative_dir
            onnx_dir.mkdir(parents=True, exist_ok=True)
            return onnx_dir / f"{self.variant.full_model_name}.onnx"
        return Path(PROJECT_ROOT_PATH) / "data" / "onnx-export" / "model.onnx"
