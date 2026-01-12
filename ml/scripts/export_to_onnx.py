from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from config.config import PROJECT_ROOT_PATH, project_config
from models.unet import UNet, UNetConfig

CHECKPOINT_FILENAME = "best_model.pth"
RESOLUTION = project_config.simulation.grid_resolution
INPUT_CHANNELS = project_config.simulation.input_channels
BATCH_SIZE = 1
ONNX_OPSET_VERSION = 18

EXPORT_FP16 = True   # GPU
EXPORT_INT8 = True   # CPU
FP16_SUFFIX = "_fp16"
INT8_SUFFIX = "_int8"


def get_export_metadata_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix('.onnx.meta')


def needs_export(checkpoint_path: Path, onnx_path: Path, force: bool) -> bool:
    """Check if export is needed - either checkpoint is newer or variants are missing."""
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

        # If checkpoint is newer than last export, re-export everything
        checkpoint_mtime = checkpoint_path.stat().st_mtime
        if checkpoint_mtime > metadata.get('checkpoint_mtime', 0):
            return True

        variants = metadata.get('variants', {})

        if EXPORT_FP16:
            fp16_path = onnx_path.with_stem(onnx_path.stem + FP16_SUFFIX)
            if 'fp16' not in variants or not fp16_path.exists():
                return True  # Need to export FP16 variant

        if EXPORT_INT8:
            int8_path = onnx_path.with_stem(onnx_path.stem + INT8_SUFFIX)
            if 'int8' not in variants or not int8_path.exists():
                return True  # Need to export INT8 variant

        return False  # All variants exist and are up-to-date

    except (json.JSONDecodeError, KeyError, OSError):
        return True  # Error reading metadata, need to re-export


def save_export_metadata(checkpoint_path: Path, base_onnx_path: Path,
                        variants_info: dict[str, dict]) -> None:
    """
    Save metadata for base model and all variants.
    variants_info = {
        'fp32': {'path': '...', 'size_mb': ..., 'mtime': ...},
        'fp16': {'path': '...', 'size_mb': ..., 'mtime': ...},
        'int8': {'path': '...', 'size_mb': ..., 'mtime': ...},
    }
    """
    metadata = {
        'checkpoint_path': str(checkpoint_path.absolute()),
        'checkpoint_mtime': checkpoint_path.stat().st_mtime,
        'export_date': datetime.now().isoformat(),
        'variants': variants_info
    }

    metadata_path = get_export_metadata_path(base_onnx_path)
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


def convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    fp16_size = fp16_path.stat().st_size / (1024 * 1024)
    reduction = ((fp32_size - fp16_size) / fp32_size) * 100

    print(f"  Converted to {fp16_path.name} ({fp16_size:.1f} MB, {reduction:.1f}% reduction)")


def quantize_model_to_int8(fp32_path: Path, int8_path: Path) -> None:
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8
    )

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    reduction = ((fp32_size - int8_size) / fp32_size) * 100

    print(f"  Quantized to {int8_path.name} ({int8_size:.1f} MB, {reduction:.1f}% reduction)")


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
    print(f"  ONNX base: {onnx_path}")

    variants_info = {}

    try:
        if not onnx_path.exists():
            print(f"  Exporting FP32...")
            model = load_model_from_checkpoint(checkpoint_path, device)
            export_model_to_onnx(model, onnx_path, input_shape, device)
        else:
            print(f"  FP32 exists, skipping")

        if onnx_path.exists():
            variants_info['fp32'] = {
                'path': str(onnx_path.absolute()),
                'size_mb': round(onnx_path.stat().st_size / (1024 * 1024), 2),
                'mtime': onnx_path.stat().st_mtime
            }

        fp16_path = onnx_path.with_stem(onnx_path.stem + FP16_SUFFIX)
        if EXPORT_FP16 and not fp16_path.exists():
            try:
                print(f"  Converting to FP16...")
                convert_to_fp16(onnx_path, fp16_path)
                variants_info['fp16'] = {
                    'path': str(fp16_path.absolute()),
                    'size_mb': round(fp16_path.stat().st_size / (1024 * 1024), 2),
                    'mtime': fp16_path.stat().st_mtime
                }
            except Exception as e:
                print(f"  Warning: FP16 conversion failed - {e}")
        elif fp16_path.exists():
            print(f"  FP16 exists, skipping")
            variants_info['fp16'] = {
                'path': str(fp16_path.absolute()),
                'size_mb': round(fp16_path.stat().st_size / (1024 * 1024), 2),
                'mtime': fp16_path.stat().st_mtime
            }

        int8_path = onnx_path.with_stem(onnx_path.stem + INT8_SUFFIX)
        if EXPORT_INT8 and not int8_path.exists():
            try:
                print(f"  Quantizing to INT8...")
                quantize_model_to_int8(onnx_path, int8_path)
                variants_info['int8'] = {
                    'path': str(int8_path.absolute()),
                    'size_mb': round(int8_path.stat().st_size / (1024 * 1024), 2),
                    'mtime': int8_path.stat().st_mtime
                }
            except Exception as e:
                print(f"  Warning: INT8 quantization failed - {e}")
        elif int8_path.exists():
            print(f"  INT8 exists, skipping")
            variants_info['int8'] = {
                'path': str(int8_path.absolute()),
                'size_mb': round(int8_path.stat().st_size / (1024 * 1024), 2),
                'mtime': int8_path.stat().st_mtime
            }

        save_export_metadata(checkpoint_path, onnx_path, variants_info)

        return True

    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
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
