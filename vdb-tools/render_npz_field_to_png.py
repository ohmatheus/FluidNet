import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from config import _PROJECT_ROOT, project_config

# Available fields that can be rendered from NPZ files
AVAILABLE_FIELDS = ["density", "velx", "velz", "vel_magnitude", "emitter", "collider"]

INPUT_NPZ = _PROJECT_ROOT / project_config.vdb_tools.npz_output_directory / "seq_0001.npz"
OUTPUT_DIR = _PROJECT_ROOT / "data/npz_image_debug/"


def _ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    # Normalize to 0..255 per array; handle constant arrays
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - mn) / (mx - mn)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _save_png(array2d: np.ndarray, path: str) -> None:
    u8 = _to_uint8(array2d)
    Image.fromarray(u8, mode="L").save(path)


def render_npz_field(
    npz_path: Path, out_dir: Path, field: str = "density", prefix: str | None = None, scale: int = 4
) -> int:
    """
    Render a field from NPZ file to PNG images.
    """
    if field not in AVAILABLE_FIELDS:
        raise ValueError(f"Unknown field: {field}. Must be one of: {', '.join(AVAILABLE_FIELDS)}")

    with np.load(npz_path) as data:
        # Load the requested field
        if field == "density":
            field_data = data["density"]  # (T,H,W)
            emitter_data = data["emitter"]  # (T,H,W) - always load emitter for density overlay
            collider_data = data.get("collider", None)  # (T,H,W) or None - load collider if available
        elif field == "velx":
            field_data = data["velx"]  # (T,H,W)
        elif field == "velz":
            field_data = data["velz"]  # (T,H,W)
        elif field == "vel_magnitude":
            # Compute velocity magnitude from components
            velx = data["velx"]
            velz = data["velz"]
            field_data = np.sqrt(velx**2 + velz**2)  # (T,H,W)
        elif field == "emitter":
            field_data = data["emitter"]  # (T,H,W)
        elif field == "collider":
            field_data = data["collider"]  # (T,H,W)
        else:
            raise ValueError(f"Unknown field: {field}")

        if field_data.ndim != 3:
            raise ValueError(f"Expected {field} to be (T,H,W), got {field_data.shape} in {npz_path}")

    T: int = int(field_data.shape[0])
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    seq_name = prefix if prefix is not None else npz_path.stem

    # Create nested directory structure: seq_name/field_name/
    seq_field_dir = out_dir / seq_name / field
    _ensure_dir(seq_field_dir)

    # Special handling for density - render as RGB with green emitter overlay
    if field == "density":
        # Normalize density to 0-255
        d_min = float(field_data.min())
        d_max = float(field_data.max())
        span = d_max - d_min
        if span <= 0:
            density_norm = np.zeros_like(field_data, dtype=np.uint8)
        else:
            density_norm = np.clip(np.round((field_data - d_min) / span * 255.0), 0, 255).astype(np.uint8)

        for t in range(T):
            fname = f"frame_{t:04d}.png"
            fpath = seq_field_dir / fname

            # Create RGB image: grayscale density + overlays
            # Start with grayscale density (R=G=B)
            rgb = np.stack([density_norm[t], density_norm[t], density_norm[t]], axis=-1)  # (H,W,3)

            # Overlay emitter in green
            emitter_mask = emitter_data[t] > 0.5  # Binary mask - useless
            rgb[emitter_mask, 0] = 0
            rgb[emitter_mask, 1] = 255
            rgb[emitter_mask, 2] = 0

            # Overlay collider in red
            if collider_data is not None:
                collider_mask = collider_data[t] > 0.5  # Binary mask - useless
                rgb[collider_mask, 0] = 255
                rgb[collider_mask, 1] = 0
                rgb[collider_mask, 2] = 0

            # Scale image
            if scale and scale != 1:
                try:
                    resample = Image.Resampling.NEAREST
                except Exception:
                    resample = Image.Resampling.NEAREST
                img = Image.fromarray(rgb, mode="RGB")
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=resample)
                img.save(fpath)
            else:
                img = Image.fromarray(rgb, mode="RGB")
                img.save(fpath)
    else:
        # Standard grayscale rendering for other fields
        # Use sequence-wide normalization for consistency across frames
        d_min = float(field_data.min())
        d_max = float(field_data.max())
        span = d_max - d_min
        if span <= 0:
            norm = np.zeros_like(field_data, dtype=np.uint8)
        else:
            norm = np.clip(np.round((field_data - d_min) / span * 255.0), 0, 255).astype(np.uint8)

        for t in range(T):
            fname = f"frame_{t:04d}.png"
            fpath = seq_field_dir / fname
            img = Image.fromarray(norm[t], mode="L")
            if scale and scale != 1:
                try:
                    # Pillow >= 10 uses Image.Resampling
                    resample = Image.Resampling.NEAREST
                except Exception:
                    resample = Image.Resampling.NEAREST
                w, h = img.size
                img = img.resize((w * scale, h * scale), resample=resample)
                img.save(fpath)
            else:
                _save_png(field_data[t], str(fpath))

    return T


def render_npz_all_fields(npz_path: Path, out_dir: Path, prefix: str | None = None, scale: int = 4) -> dict[str, int]:
    """
    Render all available fields from NPZ file to PNG images.
    """
    results = {}
    for field in AVAILABLE_FIELDS:
        try:
            n_frames = render_npz_field(npz_path, out_dir, field=field, prefix=prefix, scale=scale)
            results[field] = n_frames
            print(f"  Rendered {n_frames} {field} frames")
        except Exception as e:
            print(f"  Warning: Failed to render {field}: {e}")
            results[field] = 0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fields from seq_*.npz to PNG images (grayscale)")
    parser.add_argument(
        "--field",
        type=str,
        default="density",
        choices=AVAILABLE_FIELDS + ["all"],
        help=f"Field to render: {', '.join(AVAILABLE_FIELDS)}, or 'all' (default: density)",
    )
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for output images (default: 4)")
    args = parser.parse_args()

    input_path = Path(INPUT_NPZ)
    output_path = Path(OUTPUT_DIR)
    _ensure_dir(output_path)

    processed = 0
    render_all = args.field == "all"

    if input_path.is_dir():
        files: list[Path] = sorted(
            [f for f in input_path.iterdir() if f.name.startswith("seq_") and f.name.endswith(".npz")]
        )
        if not files:
            raise FileNotFoundError(f"No seq_*.npz found in {args.input}")
        for fp in files:
            if render_all:
                print(f"Rendering all fields from {fp}")
                render_npz_all_fields(fp, output_path, scale=args.scale)
            else:
                n = render_npz_field(fp, output_path, field=args.field, scale=args.scale)
                print(f"Rendered {n} {args.field} frames from {fp}")
            processed += 1
    else:
        if not input_path.is_file():
            raise FileNotFoundError(args.input)
        if render_all:
            print(f"Rendering all fields from {input_path}")
            render_npz_all_fields(input_path, output_path, scale=args.scale)
        else:
            n = render_npz_field(input_path, output_path, field=args.field, scale=args.scale)
            print(f"Rendered {n} {args.field} frames from {input_path}")
        processed = 1

    field_desc = "all fields" if render_all else f"field: {args.field}"
    print(f"Done. Processed {processed} sequence(s). {field_desc}. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
