from pathlib import Path

import torch
import torch.nn as nn

from config.config import PROJECT_ROOT_PATH, project_config
from models.unet import UNet, UNetConfig

CHECKPOINT_FILENAME = "best_model.pth"
RESOLUTION = project_config.simulation.grid_resolution
INPUT_CHANNELS = project_config.simulation.input_channels
BATCH_SIZE = 1
ONNX_OPSET_VERSION = 18  # Modern opset with good compatibility
DEVICE = "cuda"


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'config' key at {checkpoint_path}")

    config = checkpoint["config"]

    required_keys = ["in_channels", "out_channels", "base_channels", "depth"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Checkpoint config missing required keys: {missing_keys}")

    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    base_channels = config["base_channels"]
    depth = config["depth"]
    norm = config.get("norm", "group")
    act = config.get("act", "silu")
    group_norm_groups = config.get("group_norm_groups", 8)
    dropout = config.get("dropout", 0.0)
    upsample = config.get("upsample", "nearest")
    use_residual = config.get("use_residual", True)
    bottleneck_blocks = config.get("bottleneck_blocks", 1)

    model = UNet(
        cfg=UNetConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            norm=norm,
            act=act,
            group_norm_groups=group_norm_groups,
            dropout=dropout,
            upsample=upsample,
            use_residual=use_residual,
            bottleneck_blocks=bottleneck_blocks,
        )
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"  Loaded model from {checkpoint_path}")
    print(
        f"  Architecture: in_channels={in_channels}, out_channels={out_channels}, "
        f"base_channels={base_channels}, depth={depth}"
    )

    return model


def export_model_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, int, int, int],
    device: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed input size for now
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported to {output_path} ({file_size_mb:.1f} MB)")


def find_models_to_export(checkpoint_dir: Path, checkpoint_filename: str) -> list[tuple[str, Path]]:
    models_to_export: list[tuple[str, Path]] = []

    if not checkpoint_dir.exists():
        print(f"Warning: Checkpoint directory does not exist: {checkpoint_dir}")
        return models_to_export

    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir():
            continue

        checkpoint_path = subdir / checkpoint_filename
        if checkpoint_path.exists():
            model_name = subdir.name
            models_to_export.append((model_name, checkpoint_path))

    print(f"Found {len(models_to_export)} model(s) to export")
    return models_to_export


def main() -> None:
    checkpoint_dir = PROJECT_ROOT_PATH / project_config.models.pytorch_folder
    output_dir = PROJECT_ROOT_PATH / project_config.models.onnx_folder

    models_to_export = find_models_to_export(checkpoint_dir, CHECKPOINT_FILENAME)

    if not models_to_export:
        print("No models found to export. Exiting.")
        return

    success_count = 0
    failure_count = 0

    for model_name, checkpoint_path in models_to_export:
        print(f"\nExporting {model_name}...")

        try:
            model = load_model_from_checkpoint(checkpoint_path, DEVICE)

            output_path = output_dir / f"{model_name}.onnx"

            input_shape = (BATCH_SIZE, INPUT_CHANNELS, RESOLUTION, RESOLUTION)
            export_model_to_onnx(model, output_path, input_shape, DEVICE)

            success_count += 1

        except FileNotFoundError as e:
            print(f"  Error: Checkpoint file not found - {e}")
            failure_count += 1

        except KeyError as e:
            print(f"  Error: Missing configuration in checkpoint - {e}")
            failure_count += 1

        except RuntimeError as e:
            print(f"  Error: ONNX export failed - {e}")
            failure_count += 1

        except Exception as e:
            print(f"  Error: Unexpected error - {type(e).__name__}: {e}")
            failure_count += 1

    print(f"\nExport complete: {success_count}/{len(models_to_export)} models exported successfully")
    if failure_count > 0:
        print(f"  {failure_count} model(s) failed to export")


if __name__ == "__main__":
    main()
