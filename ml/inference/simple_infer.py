import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.small_unet import SmallUNet
from config.config import ml_config, PROJECT_ROOT_PATH, ML_ROOT_PATH


def load_model(checkpoint_path: str | Path, device: str) -> SmallUNet:
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

    model = SmallUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Architecture: in_channels={in_channels}, out_channels={out_channels}, "
          f"base_channels={base_channels}, depth={depth}")

    return model


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


def main():
    MODEL_NAME = "SmallUnet"
    CHECKPOINT_PATH = Path(PROJECT_ROOT_PATH / f"data/checkpoints/{MODEL_NAME}/final_model.pth")
    OUTPUT_DIR = Path(PROJECT_ROOT_PATH / "data/simple_infer_output")
    NUM_FRAMES = 600
    RESOLUTION = 128
    UPSCALE_FACTOR = 4
    USE_GLOBAL_NORM = True
    INJECTED_DENSITY = 0.8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(CHECKPOINT_PATH, device)

    # Initialize state buffers with circle density (previous and current frame)
    state_prev = torch.zeros(3, RESOLUTION, RESOLUTION, device=device)
    state_current = torch.zeros(3, RESOLUTION, RESOLUTION, device=device)

    circle_radius = 20
    center_x = RESOLUTION // 2
    center_y = RESOLUTION // 2 + 30 # Near bottom

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(RESOLUTION, device=device),
        torch.arange(RESOLUTION, device=device),
        indexing='ij'
    )

    dist = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create circular mask - This is ducktape debugging
    circle_mask = dist <= circle_radius

    # Set density to 1 inside the circle for both frames
    state_prev[0][circle_mask] = INJECTED_DENSITY
    state_current[0][circle_mask] = INJECTED_DENSITY

    print(f"\nStarting autoregressive rollout for {NUM_FRAMES} frames...")
    print(f"Initial condition: circle of density=1 (radius={circle_radius}) at center-bottom ({center_x}, {center_y})")

    density_frames = []

    # Autoregressive rollout
    with torch.no_grad():
        for frame_idx in range(NUM_FRAMES):
            # [density_t, velx_t, vely_t, density_{t-1}]
            # state_current: [density_t, velx_t, vely_t]
            # state_prev: [density_{t-1}, velx_{t-1}, vely_{t-1}]
            
            # fake injecting density like in blender sim
            state_prev[0][circle_mask] += INJECTED_DENSITY
            state_current[0][circle_mask] += INJECTED_DENSITY
            
            model_input = torch.cat([
                state_current,           # [density_t, velx_t, vely_t]
                state_prev[0:1]         # [density_{t-1}]
            ], dim=0)  # Shape: (4, H, W)

            # Add batch dimension
            model_input = model_input.unsqueeze(0)  # Shape: (1, 4, H, W)

            output = model(model_input)  # Shape: (1, 3, H, W)

            # Remove batch dimension
            output = output.squeeze(0)  # Shape: (3, H, W)

            # Extract density for visualization
            density = output[0].cpu().numpy()  # Shape: (H, W)
            density_frames.append(density)

            # Update buffers for next iteration
            state_prev = state_current.clone()
            state_current = output

            if (frame_idx + 1) % 10 == 0:
                print(f"  Generated frame {frame_idx + 1}/{NUM_FRAMES}")

    print(f"\nGeneration complete. Saving visualizations...")

    # Render and save all frames as png
    render_density_to_png(
        density_frames,
        OUTPUT_DIR,
        scale=UPSCALE_FACTOR,
        use_global_norm=USE_GLOBAL_NORM,
    )

    print(f"\nDone! Output saved to: {OUTPUT_DIR.resolve()}")
    print(f"Generated {NUM_FRAMES} frames at {RESOLUTION}x{RESOLUTION}, upscaled {UPSCALE_FACTOR}x")


if __name__ == "__main__":
    main()
