import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Available fields that can be rendered from NPZ files
AVAILABLE_FIELDS = ["density", "velx", "velz", "vel_magnitude"]


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
        elif field == "velx":
            field_data = data["velx"]  # (T,H,W)
        elif field == "velz":
            field_data = data["velz"]  # (T,H,W)
        elif field == "vel_magnitude":
            # Compute velocity magnitude from components
            velx = data["velx"]
            velz = data["velz"]
            field_data = np.sqrt(velx**2 + velz**2)  # (T,H,W)

        if field_data.ndim != 3:
            raise ValueError(f"Expected {field} to be (T,H,W), got {field_data.shape} in {npz_path}")

    T: int = int(field_data.shape[0])
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    seq_name = prefix if prefix is not None else npz_path.stem

    # Create nested directory structure: seq_name/field_name/
    seq_field_dir = out_dir / seq_name / field
    _ensure_dir(seq_field_dir)

    # Use sequence-wide normalization for consistency across frames
    # For velocity components (can be negative), we want to handle the full range
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
                resample = Image.Resampling.BILINEAR  # getattr(Image, "Resampling", Image).BILINEAR
            except Exception:
                resample = Image.Resampling.BILINEAR
            w, h = img.size
            img = img.resize((w * scale, h * scale), resample=resample)
            img.save(fpath)
        else:
            # Per-frame fallback conversion
            _save_png(field_data[t], fpath)

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
    parser.add_argument("input", type=str, help="Path to a seq_*.npz file or a directory containing them")
    parser.add_argument("output", type=str, help="Directory to write images")
    parser.add_argument(
        "--field",
        type=str,
        default="density",
        choices=AVAILABLE_FIELDS + ["all"],
        help=f"Field to render: {', '.join(AVAILABLE_FIELDS)}, or 'all' (default: density)",
    )
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for output images (default: 4)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
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
    print(f"Done. Processed {processed} sequence(s). {field_desc}. Output: {args.output}")


if __name__ == "__main__":
    main()
