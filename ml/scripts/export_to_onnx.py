from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn

from config.config import PROJECT_ROOT_PATH, project_config
from models.unet import UNet, UNetConfig

CHECKPOINT_FILENAME = "best_model.pth"
RESOLUTION = project_config.simulation.grid_resolution
INPUT_CHANNELS = project_config.simulation.input_channels
BATCH_SIZE = 1
ONNX_OPSET_VERSION = 18


def get_export_metadata_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix('.onnx.meta')


def needs_export(checkpoint_path: Path, onnx_path: Path, force: bool) -> bool:
    if force:
        return True

    if not onnx_path.exists():
        return True

    metadata_path = get_export_metadata_path(onnx_path)
    if not metadata_path.exists():
        return True

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        checkpoint_mtime = checkpoint_path.stat().st_mtime
        stored_mtime = metadata.get('checkpoint_mtime', 0)

        if checkpoint_mtime > stored_mtime:
            return True

        return False
    except (json.JSONDecodeError, KeyError, OSError):
        return True


def save_export_metadata(checkpoint_path: Path, onnx_path: Path) -> None:
    metadata = {
        'checkpoint_path': str(checkpoint_path.absolute()),
        'checkpoint_mtime': checkpoint_path.stat().st_mtime,
        'onnx_path': str(onnx_path.absolute()),
        'onnx_mtime': onnx_path.stat().st_mtime,
        'export_date': datetime.now().isoformat()
    }

    metadata_path = get_export_metadata_path(onnx_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


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
    padding_mode = config.get("padding_mode", "zeros")
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
            padding_mode=padding_mode,
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


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export PyTorch checkpoints to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for export (default: cuda if available, else cpu)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export even if ONNX file is up-to-date (ignores metadata)"
    )

    return parser


def discover_checkpoints_to_export(checkpoints_dir: Path, onnx_export_dir: Path) -> list[tuple[str, Path, Path]]:
    checkpoints_to_export: list[tuple[str, Path, Path]] = []

    if not checkpoints_dir.exists():
        print(f"Warning: Checkpoints directory does not exist: {checkpoints_dir}")
        return checkpoints_to_export

    # Recursively find all best_model.pth files
    for checkpoint_file in checkpoints_dir.rglob(CHECKPOINT_FILENAME):
        model_folder = checkpoint_file.parent

        relative_path = model_folder.relative_to(checkpoints_dir)

        model_name = model_folder.name
        onnx_path = onnx_export_dir / relative_path / f"{model_name}.onnx"

        display_name = str(relative_path / model_name)

        checkpoints_to_export.append((display_name, checkpoint_file, onnx_path))

    checkpoints_to_export.sort(key=lambda x: x[0])

    return checkpoints_to_export


def export_checkpoint(display_name: str, checkpoint_path: Path, onnx_path: Path, device: str, input_shape: tuple[int, int, int, int]) -> bool:
    print(f"\nExporting: {display_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  ONNX output: {onnx_path}")

    try:
        model = load_model_from_checkpoint(checkpoint_path, device)
        export_model_to_onnx(model, onnx_path, input_shape, device)
        save_export_metadata(checkpoint_path, onnx_path)
        return True

    except KeyError as e:
        print(f"  Error: Missing configuration in checkpoint - {e}")
        return False
    except RuntimeError as e:
        print(f"  Error: ONNX export failed - {e}")
        return False
    except Exception as e:
        print(f"  Error: Unexpected error - {type(e).__name__}: {e}")
        return False


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    checkpoints_dir = PROJECT_ROOT_PATH / "data" / "checkpoints"
    onnx_export_dir = PROJECT_ROOT_PATH / "data" / "onnx-export"

    print("Scanning checkpoints directory...\n")
    checkpoints_to_export = discover_checkpoints_to_export(checkpoints_dir, onnx_export_dir)

    if not checkpoints_to_export:
        print("No checkpoints found to export. Exiting.")
        return

    print(f"Found {len(checkpoints_to_export)} checkpoint(s) to process\n")

    success_count = 0
    failure_count = 0
    skipped_count = 0
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, RESOLUTION, RESOLUTION)

    for display_name, checkpoint_path, onnx_path in checkpoints_to_export:
        if not args.force and not needs_export(checkpoint_path, onnx_path, args.force):
            print(f"Skipping {display_name} (up-to-date)")
            skipped_count += 1
            continue

        if export_checkpoint(display_name, checkpoint_path, onnx_path, args.device, input_shape):
            success_count += 1
        else:
            failure_count += 1

    print(f"\n{'='*60}")
    print(f"Export Summary:")
    print(f"  Exported: {success_count}/{len(checkpoints_to_export)}")
    print(f"  Skipped (up-to-date): {skipped_count}")
    if failure_count > 0:
        print(f"  Failed: {failure_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
