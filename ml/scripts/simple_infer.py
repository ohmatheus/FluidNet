import argparse
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from PIL import Image

from config.config import PROJECT_ROOT_PATH
from inference.backend import InferenceBackend, ONNXBackend, PyTorchBackend


def log_channel_stats(frame_idx: int, data: np.ndarray, label: str, log_file: TextIO) -> None:
    density, velx, vely = data[0], data[1], data[2]
    log_file.write(f"[Frame {frame_idx:04d}] {label}:\n")
    log_file.write(f"  density: min={density.min():.6f}, max={density.max():.6f}\n")
    log_file.write(f"  velx:    min={velx.min():.6f}, max={velx.max():.6f}\n")
    log_file.write(f"  vely:    min={vely.min():.6f}, max={vely.max():.6f}\n")
    log_file.flush()  # Ensure logs are written immediately


def render_density_to_png(
    density_fields: list[np.ndarray],
    output_dir: Path,
    scale: int = 4,
    use_global_norm: bool = False,
    collider_mask: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_global_norm:
        # Global normalization: find min/max across all frames
        all_density = np.stack(density_fields, axis=0)  # (T, H, W)
        d_min = float(all_density.min())
        d_max = float(all_density.max())
        span = d_max - d_min

        if span > 0:
            # Normalize all frames together
            norm_all = np.clip(np.round((all_density - d_min) / span * 255.0), 0, 255).astype(np.uint8)
        else:
            norm_all = np.zeros_like(all_density, dtype=np.uint8)

        # Save each frame
        for i, frame in enumerate(norm_all):
            img = Image.fromarray(frame, mode="L").convert("RGB")
            if collider_mask is not None:
                pixels = np.array(img)
                pixels[collider_mask > 0] = [255, 0, 0]
                img = Image.fromarray(pixels)
            if scale > 1:
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=Image.Resampling.BILINEAR)

            fname = f"frame_{i:04d}.png"
            img.save(output_dir / fname)

        print(f"Saved {len(density_fields)} frames with global normalization (range: [{d_min:.4f}, {d_max:.4f}])")
    else:
        # No normalization - direct clipping to [0, 255]
        for i, density in enumerate(density_fields):
            norm = np.clip(np.round(density * 255.0), 0, 255).astype(np.uint8)

            img = Image.fromarray(norm, mode="L").convert("RGB")
            if collider_mask is not None:
                pixels = np.array(img)
                pixels[collider_mask > 0] = [255, 0, 0]
                img = Image.fromarray(pixels)
            if scale > 1:
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=Image.Resampling.BILINEAR)

            fname = f"frame_{i:04d}.png"
            img.save(output_dir / fname)

        print(f"Saved {len(density_fields)} frames without normalization (direct clipping)")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI options."""
    parser = argparse.ArgumentParser(description="Run inference on fluid dynamics model")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "onnx"],
        default="pytorch",
        help="Backend to use for inference (default: pytorch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="SmallUnetFull",
        help="Model name (checkpoint folder name, default: SmallUnetFull)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=600,
        help="Number of frames to generate (default: 600)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Simulation resolution (H x W, default: 128)",
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=4,
        help="Upscale factor for output images (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rendered frames (default: data/simple_infer_output)",
    )
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()

    USE_GLOBAL_NORM = False

    if args.backend == "pytorch":
        backend: InferenceBackend = PyTorchBackend()
        model_path = PROJECT_ROOT_PATH / f"data/checkpoints/{args.model_name}/best_model.pth"
    else:  # onnx
        backend = ONNXBackend()
        model_path = PROJECT_ROOT_PATH / f"data/onnx/{args.model_name}.onnx"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"For PyTorch backend, ensure checkpoint exists in data/checkpoints/{args.model_name}/\n"
            f"For ONNX backend, run 'onnx-export' first to create the ONNX model."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT_PATH / "data/simple_infer_output"

    print(f"Using backend: {args.backend}")
    print(f"Using device: {args.device}")

    backend.load_model(model_path, args.device)

    # Initialize state buffers with circle density (previous and current frame)
    state_prev = np.zeros((3, args.resolution, args.resolution), dtype=np.float32)
    state_current = np.zeros((3, args.resolution, args.resolution), dtype=np.float32)

    circle_radius = 10
    center_x = args.resolution // 2
    center_y = args.resolution // 2 + 50  # Near bottom

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(args.resolution), np.arange(args.resolution), indexing="ij")

    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create circular mask
    circle_mask = dist <= circle_radius

    # Create binary emitter mask (1 inside circle, 0 outside)
    # This is a static mask - same for all frames
    emitter_mask = np.zeros((args.resolution, args.resolution), dtype=np.float32)
    emitter_mask[circle_mask] = 1.0

    # Create rectangular collider mask at middle top
    collider_width = 40  # Rectangle width
    collider_height = 8  # Rectangle height
    collider_center_x = args.resolution // 2
    collider_center_y = args.resolution // 2 - 40

    # Create rectangular mask (1 inside rectangle, 0 outside)
    collider_mask = np.zeros((args.resolution, args.resolution), dtype=np.float32)
    x_start = max(0, collider_center_x - collider_width // 2)
    x_end = min(args.resolution, collider_center_x + collider_width // 2)
    y_start = max(0, collider_center_y - collider_height // 2)
    y_end = min(args.resolution, collider_center_y + collider_height // 2)
    collider_mask[y_start:y_end, x_start:x_end] = 1.0

    # Initialize density=1.0 in emitter region (but not where collider is)
    initial_density_mask = (emitter_mask > 0) & (collider_mask == 0)
    state_prev[0] = initial_density_mask.astype(np.float32)
    state_current[0] = initial_density_mask.astype(np.float32)

    print(f"\nStarting autoregressive rollout for {args.num_frames} frames...")
    print("Emitter mask: binary circle (1 inside, 0 outside) - static for all frames")
    print("Collider mask: binary rectangle (1 inside, 0 outside) - static for all frames")
    print(f"Initial density: 1.0 in emitter (excluding collider region), 0.0 elsewhere")

    density_frames = []

    # Create log file for debug output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "inference_debug.txt"
    log_file = open(log_file_path, "w")
    print(f"Debug logs will be saved to: {log_file_path.resolve()}")

    # Autoregressive rollout
    for frame_idx in range(args.num_frames):
        emitter_channel = emitter_mask[np.newaxis, ...]  # Shape: (1, H, W)
        collider_channel = collider_mask[np.newaxis, ...]  # Shape: (1, H, W)

        model_input = np.concatenate(
            [
                state_current,  # [density_t, velx_t, vely_t]
                state_prev[0:1],  # [density_{t-1}]
                emitter_channel,  # [emitter_mask]
                collider_channel,  # [collider_mask]
            ],
            axis=0,
        )  # Shape: (6, H, W)

        # Log input statistics (first 3 channels: density, velx, vely)
        log_channel_stats(frame_idx, model_input[:3], "INPUT", log_file)

        # Add batch dimension
        model_input = model_input[np.newaxis, ...]  # Shape: (1, 5, H, W)

        output = backend.infer(model_input)  # Shape: (1, 3, H, W)

        # Remove batch dimension
        output = output[0]  # Shape: (3, H, W)

        # Log output statistics
        log_channel_stats(frame_idx, output, "OUTPUT", log_file)

        # Extract density for visualization
        density = output[0]  # Shape: (H, W)
        #vel_mag = np.sqrt(output[1]**2 + output[2]**2)
        #vel_mag = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-8)

        density_frames.append(density)

        # Update buffers for next iteration
        state_prev = state_current.copy()
        state_current = output

        if (frame_idx + 1) % 10 == 0:
            print(f"  Generated frame {frame_idx + 1}/{args.num_frames}")

    log_file.close()

    print("\nGeneration complete. Saving visualizations...")

    render_density_to_png(
        density_frames,
        output_dir,
        scale=args.upscale_factor,
        use_global_norm=USE_GLOBAL_NORM,
        collider_mask=collider_mask,
    )

    print(f"\nDone! Output saved to: {output_dir.resolve()}")
    print(f"Generated {args.num_frames} frames at {args.resolution}x{args.resolution}, upscaled {args.upscale_factor}x")


if __name__ == "__main__":
    main()
