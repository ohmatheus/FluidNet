import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config.config import PROJECT_ROOT_PATH
from inference.backend import InferenceBackend, ONNXBackend, PyTorchBackend


def render_density_to_png(
    density_fields: list[np.ndarray],
    output_dir: Path,
    scale: int = 4,
    use_global_norm: bool = True,
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
            img = Image.fromarray(frame, mode="L")
            if scale > 1:
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=Image.Resampling.BILINEAR)

            fname = f"frame_{i:04d}.png"
            img.save(output_dir / fname)

        print(f"Saved {len(density_fields)} frames with global normalization (range: [{d_min:.4f}, {d_max:.4f}])")
    else:
        # Per-frame normalization
        for i, density in enumerate(density_fields):
            d_min = float(density.min())
            d_max = float(density.max())
            span = d_max - d_min

            if span > 0:
                norm = np.clip(np.round((density - d_min) / span * 255.0), 0, 255).astype(np.uint8)
            else:
                norm = np.zeros_like(density, dtype=np.uint8)

            img = Image.fromarray(norm, mode="L")
            if scale > 1:
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=Image.Resampling.BILINEAR)

            fname = f"frame_{i:04d}.png"
            img.save(output_dir / fname)

        print(f"Saved {len(density_fields)} frames with per-frame normalization")


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
        default="SmallUnet",
        help="Model name (checkpoint folder name, default: SmallUnet)",
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

    INJECTED_DENSITY = 0.8
    USE_GLOBAL_NORM = True

    if args.backend == "pytorch":
        backend: InferenceBackend = PyTorchBackend()
        model_path = PROJECT_ROOT_PATH / f"data/checkpoints/{args.model_name}/final_model.pth"
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

    circle_radius = 20
    center_x = args.resolution // 2
    center_y = args.resolution // 2 + 30  # Near bottom

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(args.resolution), np.arange(args.resolution), indexing="ij")

    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create circular mask
    circle_mask = dist <= circle_radius

    # Set density inside the circle for both frames
    state_prev[0][circle_mask] = INJECTED_DENSITY
    state_current[0][circle_mask] = INJECTED_DENSITY

    print(f"\nStarting autoregressive rollout for {args.num_frames} frames...")
    print(
        f"Initial condition: circle of density={INJECTED_DENSITY} (radius={circle_radius}) at ({center_x}, {center_y})"
    )

    density_frames = []

    # Autoregressive rollout
    for frame_idx in range(args.num_frames):
        # [density_t, velx_t, vely_t, density_{t-1}]
        # state_current: [density_t, velx_t, vely_t]
        # state_prev: [density_{t-1}, velx_{t-1}, vely_{t-1}]

        # Inject density like in blender sim
        state_prev[0][circle_mask] += INJECTED_DENSITY
        state_current[0][circle_mask] += INJECTED_DENSITY

        # Build model input
        model_input = np.concatenate(
            [
                state_current,  # [density_t, velx_t, vely_t]
                state_prev[0:1],  # [density_{t-1}]
            ],
            axis=0,
        )  # Shape: (4, H, W)

        # Add batch dimension
        model_input = model_input[np.newaxis, ...]  # Shape: (1, 4, H, W)

        output = backend.infer(model_input)  # Shape: (1, 3, H, W)

        # Remove batch dimension
        output = output[0]  # Shape: (3, H, W)

        # Extract density for visualization
        density = output[0]  # Shape: (H, W)
        density_frames.append(density)

        # Update buffers for next iteration
        state_prev = state_current.copy()
        state_current = output

        if (frame_idx + 1) % 10 == 0:
            print(f"  Generated frame {frame_idx + 1}/{args.num_frames}")

    print("\nGeneration complete. Saving visualizations...")

    render_density_to_png(
        density_frames,
        output_dir,
        scale=args.upscale_factor,
        use_global_norm=USE_GLOBAL_NORM,
    )

    print(f"\nDone! Output saved to: {output_dir.resolve()}")
    print(f"Generated {args.num_frames} frames at {args.resolution}x{args.resolution}, upscaled {args.upscale_factor}x")


if __name__ == "__main__":
    main()
