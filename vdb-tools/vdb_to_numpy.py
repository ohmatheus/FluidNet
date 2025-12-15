import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import openvdb  # type: ignore[import-not-found]
from scipy.ndimage import zoom  # type: ignore[import-untyped]


def extract_frame_number(vdb_path: Path) -> int:
    vdb_path = Path(vdb_path)
    basename = vdb_path.name
    try:
        parts = basename.replace(".vdb", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    except Exception:
        return 0


def get_grid_names(vdb_path: Path) -> list[str]:
    target_grids = ["density", "velocity"]  # add later here collision mask, etc
    available_grids: list[str] = []

    vdb_path_str = str(vdb_path)

    for grid_name in target_grids:
        try:
            _ = openvdb.read(vdb_path_str, grid_name)
            available_grids.append(grid_name)
        except Exception as e:
            print(f"  Warning: Could not read grid '{grid_name}' from {Path(vdb_path).name}: {e}")
            continue

    return available_grids


def extract_velocity_components_avg(
    grid: Any, target_resolution: int, density_grid: Any = None
) -> dict[str, np.ndarray]:
    """
    Extract both X and Z velocity components by averaging across Y layers.
    If density_grid is provided, uses density-weighted averaging; otherwise uses uniform averaging.
    Pads active voxel region to full domain (target_resolution x target_resolution) before resizing.
    """
    print(f"    Processing velocity grid (type: {type(grid)})")

    try:
        bbox_result = grid.evalActiveVoxelBoundingBox()

        if isinstance(bbox_result, tuple) and len(bbox_result) == 2:
            min_tuple, max_tuple = bbox_result
            min_x, min_y, min_z = min_tuple
            max_x, max_y, max_z = max_tuple

            print(f"    Bounds: X[{min_x}:{max_x}], Y[{min_y}:{max_y}], Z[{min_z}:{max_z}]")

            dim_x = max_x - min_x + 1
            dim_y = max_y - min_y + 1
            dim_z = max_z - min_z + 1

            print(f"    Averaging across {dim_y} Y layers, dimensions (X,Z): {dim_x}x{dim_z}")

            # IMPORTANT: store arrays as (Z, X) so that rows correspond to Z (vertical) and columns to X (horizontal)
            vel_x_sum = np.zeros((dim_z, dim_x), dtype=np.float32)
            vel_z_sum = np.zeros((dim_z, dim_x), dtype=np.float32)
            weight_sum = np.zeros((dim_z, dim_x), dtype=np.float32)

            accessor = grid.getAccessor()
            dens_accessor = density_grid.getAccessor() if density_grid is not None else None
            valid_samples = 0

            # Sum across all Y layers (density-weighted if density_grid provided)
            for y in range(min_y, max_y + 1):
                for i, x in enumerate(range(min_x, max_x + 1)):
                    for j, z in enumerate(range(min_z, max_z + 1)):
                        try:
                            # Try different coordinate formats
                            try:
                                velocity_vec = accessor.getValue((x, y, z))
                            except Exception:
                                try:
                                    velocity_vec = accessor.getValue(x, y, z)
                                except Exception:
                                    continue

                            # Extract X and Z components from 3D velocity vector
                            if hasattr(velocity_vec, "__len__") and len(velocity_vec) >= 3:
                                vx = float(velocity_vec[0])  # X component
                                vz = float(velocity_vec[2])  # Z component

                                # Get weight: density value if available, else 1.0
                                if dens_accessor is not None:
                                    try:
                                        dens_val = dens_accessor.getValue((x, y, z))
                                    except Exception:
                                        try:
                                            dens_val = dens_accessor.getValue(x, y, z)
                                        except Exception:
                                            dens_val = 0.0
                                    weight = float(dens_val)
                                    if weight <= 0.0:
                                        continue  # Skip voxels with zero or negative density
                                else:
                                    weight = 1.0  # Uniform averaging fallback

                                # Accumulate weighted sums
                                vel_x_sum[j, i] += weight * vx
                                vel_z_sum[j, i] += weight * vz
                                weight_sum[j, i] += weight

                                if vx != 0.0 or vz != 0.0:
                                    valid_samples += 1

                        except Exception:
                            continue

            # Average by dividing by weight sum (avoid division by zero)
            vel_x_data = np.divide(vel_x_sum, weight_sum, out=np.zeros_like(vel_x_sum), where=weight_sum != 0)
            vel_z_data = np.divide(vel_z_sum, weight_sum, out=np.zeros_like(vel_z_sum), where=weight_sum != 0)

            print(f"    Valid velocity samples: {valid_samples}")

            # Pad to full domain size to preserve spatial relationships
            # Calculate padding to restore full domain coordinates
            # Arrays are (Z, X) format
            pad_x_before = max(0, min_x - 1)
            pad_x_after = max(0, target_resolution - max_x)
            pad_z_before = max(0, min_z - 1)
            pad_z_after = max(0, target_resolution - max_z)

            if pad_x_before > 0 or pad_x_after > 0 or pad_z_before > 0 or pad_z_after > 0:
                print(
                    f"    Padding to domain size {target_resolution}: X[{pad_x_before}, {pad_x_after}], Z[{pad_z_before}, {pad_z_after}]"
                )
                vel_x_data = np.pad(
                    vel_x_data,
                    ((pad_z_before, pad_z_after), (pad_x_before, pad_x_after)),
                    mode="constant",
                    constant_values=0,
                )
                vel_z_data = np.pad(
                    vel_z_data,
                    ((pad_z_before, pad_z_after), (pad_x_before, pad_x_after)),
                    mode="constant",
                    constant_values=0,
                )
                print(f"    Padded shape: {vel_x_data.shape}")

            # After padding, shape should already match target_resolution
            # Only resize if there's a mismatch (shouldn't happen normally)
            if vel_x_data.shape[0] != target_resolution or vel_x_data.shape[1] != target_resolution:
                print(f"    Warning: Unexpected shape {vel_x_data.shape}, resizing to {target_resolution}")
                zoom_z = target_resolution / vel_x_data.shape[0]
                zoom_x = target_resolution / vel_x_data.shape[1]

                vel_x_data = zoom(vel_x_data, (zoom_z, zoom_x), order=1)
                vel_z_data = zoom(vel_z_data, (zoom_z, zoom_x), order=1)

            # Use keys compatible with the ML dataset (velx/velz)
            return {"velx": vel_x_data, "velz": vel_z_data}

        else:
            return {
                "velx": np.zeros((target_resolution, target_resolution), dtype=np.float32),
                "velz": np.zeros((target_resolution, target_resolution), dtype=np.float32),
            }

    except Exception as e:
        print(f"    Error extracting velocity: {e}")
        return {
            "velx": np.zeros((target_resolution, target_resolution), dtype=np.float32),
            "velz": np.zeros((target_resolution, target_resolution), dtype=np.float32),
        }


def extract_density_field_sum(grid: Any, target_resolution: int) -> np.ndarray:
    """
    Extract density field by summing across Y layers.
    Pads active voxel region to full domain (target_resolution x target_resolution) before resizing.
    """
    print(f"    Processing density grid (type: {type(grid)})")

    try:
        bbox_result = grid.evalActiveVoxelBoundingBox()

        if isinstance(bbox_result, tuple) and len(bbox_result) == 2:
            min_tuple, max_tuple = bbox_result
            min_x, min_y, min_z = min_tuple
            max_x, max_y, max_z = max_tuple

            print(f"    Bounds: X[{min_x}:{max_x}], Y[{min_y}:{max_y}], Z[{min_z}:{max_z}]")

            dim_x = max_x - min_x + 1
            dim_y = max_y - min_y + 1
            dim_z = max_z - min_z + 1

            print(f"    Summing across {dim_y} Y layers, dimensions (X,Z): {dim_x}x{dim_z}")

            # Store as (Z, X) so rows=Z, cols=X
            density_data = np.zeros((dim_z, dim_x), dtype=np.float32)
            accessor = grid.getAccessor()
            valid_samples = 0

            # Sum across all Y layers
            for y in range(min_y, max_y + 1):
                for i, x in enumerate(range(min_x, max_x + 1)):
                    for j, z in enumerate(range(min_z, max_z + 1)):
                        try:
                            # Try different coordinate formats
                            try:
                                density_val = accessor.getValue((x, y, z))
                            except Exception:
                                try:
                                    density_val = accessor.getValue(x, y, z)
                                except Exception:
                                    continue

                            density_data[j, i] += float(density_val)

                            if density_val > 0.0:
                                valid_samples += 1

                        except Exception:
                            continue

            print(f"    Valid density samples: {valid_samples}")

            # Pad to full domain size to preserve spatial relationships
            # Calculate padding to restore full domain coordinates
            # Arrays are (Z, X) format
            pad_x_before = max(0, min_x - 1)
            pad_x_after = max(0, target_resolution - max_x)
            pad_z_before = max(0, min_z - 1)
            pad_z_after = max(0, target_resolution - max_z)

            if pad_x_before > 0 or pad_x_after > 0 or pad_z_before > 0 or pad_z_after > 0:
                print(
                    f"    Padding to domain size {target_resolution}: X[{pad_x_before}, {pad_x_after}], Z[{pad_z_before}, {pad_z_after}]"
                )
                density_data = np.pad(
                    density_data,
                    ((pad_z_before, pad_z_after), (pad_x_before, pad_x_after)),
                    mode="constant",
                    constant_values=0,
                )
                print(f"    Padded shape: {density_data.shape}")

            # After padding, shape should already match target_resolution
            # Only resize if there's a mismatch (shouldn't happen normally)
            if density_data.shape[0] != target_resolution or density_data.shape[1] != target_resolution:
                print(f"    Warning: Unexpected shape {density_data.shape}, resizing to {target_resolution}")
                zoom_z = target_resolution / density_data.shape[0]
                zoom_x = target_resolution / density_data.shape[1]
                density_data = zoom(density_data, (zoom_z, zoom_x), order=1)

            print(f"    Density range: [{density_data.min():.6f}, {density_data.max():.6f}]")

            return density_data

        else:
            return np.zeros((target_resolution, target_resolution), dtype=np.float32)

    except Exception as e:
        print(f"    Error extracting density: {e}")
        return np.zeros((target_resolution, target_resolution), dtype=np.float32)


def process_vdb_file(
    vdb_path: Path,
    output_dir: Path,
    target_resolution: int = 32,
    save_frames: bool = False,
    axis_order: str = "XZ",
    flip_x: bool = True,
    flip_z: bool = False,
) -> dict[str, np.ndarray]:
    """Process single VDB file - sum density, average velocity across Y."""

    vdb_path = Path(vdb_path)
    output_dir = Path(output_dir)
    print(f"\nProcessing: {vdb_path.name}")

    try:
        available_grids = get_grid_names(vdb_path)

        if not available_grids:
            print("  No density/velocity grids found")
            return {}

        print(f"  Available grids: {available_grids}")

        frame_num = extract_frame_number(vdb_path)
        frame_data = {}
        density_grid_obj = None  # Store 3D density grid for velocity weighting

        for grid_name in available_grids:
            print(f"  Processing '{grid_name}':")

            try:
                grid = openvdb.read(str(vdb_path), grid_name)

                if grid_name == "density":
                    density_grid_obj = grid  # Store 3D density grid for velocity weighting
                    density = extract_density_field_sum(grid, target_resolution)
                    # Blender extraction returns (Z, X)
                    if axis_order.upper() == "XZ":
                        density = np.swapaxes(density, 0, 1)  # (X, Z)
                    elif axis_order.upper() == "ZX":
                        pass
                    else:
                        raise ValueError("axis_order must be 'XZ' or 'ZX'")

                    # Apply flips in the chosen axis order: axis 0 is first letter, axis 1 is second.
                    if flip_x:
                        density = np.flip(density, axis=0)
                    if flip_z:
                        density = np.flip(density, axis=1)

                    frame_data["density"] = density
                    if save_frames:
                        # Save density frame as .npy when requested
                        output_path = output_dir / f"density_{frame_num:04d}.npy"
                        np.save(output_path, density)
                        print(f"    Saved: density_{frame_num:04d}.npy")

                elif grid_name == "velocity":
                    velocity_components = extract_velocity_components_avg(
                        grid, target_resolution, density_grid=density_grid_obj
                    )
                    # Reorder and flip each component the same way as density
                    # IMPORTANT: When flipping axes, velocity components must be negated for physical correctness
                    for key in ("velx", "velz"):
                        if key in velocity_components:
                            arr = velocity_components[key]
                            if axis_order.upper() == "XZ":
                                arr = np.swapaxes(arr, 0, 1)  # (X, Z)
                            elif axis_order.upper() == "ZX":
                                pass
                            else:
                                raise ValueError("axis_order must be 'XZ' or 'ZX'")

                            # Apply spatial flips and negate component signs for physical correctness
                            if flip_x:
                                arr = np.flip(arr, axis=0)
                                if key == "velx":  # Flipping X-axis: negate X-velocity component
                                    arr = -arr
                            if flip_z:
                                arr = np.flip(arr, axis=1)
                                if key == "velz":  # Flipping Z-axis: negate Z-velocity component
                                    arr = -arr

                            velocity_components[key] = arr
                    frame_data.update(velocity_components)
                    if save_frames:
                        # Save velocity components as .npy when requested
                        for comp_name, comp_data in velocity_components.items():
                            output_path = output_dir / f"{comp_name}_{frame_num:04d}.npy"
                            np.save(output_path, comp_data)
                            print(f"    Saved: {comp_name}_{frame_num:04d}.npy")

            except Exception as e:
                print(f"    Error processing '{grid_name}': {e}")

        return frame_data

    except Exception as e:
        print(f"  Error: {e}")
        return {}


def process_all_frames(
    cache_dir: Path,
    output_dir: Path,
    target_resolution: int,
    max_frames: int | None = None,
    save_frames: bool = False,
) -> None:
    """
    Process all VDB files in directory and directly pack non-overlapping sequences into seq_*.npz.
    """

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    vdb_files = sorted(cache_dir.glob("*.vdb"))

    if not vdb_files:
        print(f"No VDB files found in {cache_dir}")
        return

    print(f"Found {len(vdb_files)} VDB files")
    print(f"Target resolution: {target_resolution}x{target_resolution}")
    print("Processing: DENSITY (sum across Y), VELOCITY (average across Y)")
    # Deduce sequence length if not provided: use number of files that look like 'fluid_data_0001.vdb'
    pat = re.compile(r"^fluid_data_\d{4,}\.vdb$")
    candidate_files = [f for f in vdb_files if pat.match(f.name)]
    deduced = len(candidate_files) if candidate_files else len(vdb_files)
    seq_len = deduced
    print(f"Deduced sequence length: {seq_len} (from folder contents)")

    if max_frames:
        vdb_files = vdb_files[:max_frames]
        print(f"Processing first {len(vdb_files)} files")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    successful = 0
    total_nonzero_files = 0

    density_frames = []  # List[np.ndarray]
    velx_frames = []
    velz_frames = []
    hw_ref = None

    for _i, vdb_file in enumerate(vdb_files):
        frame_data = process_vdb_file(
            vdb_file,
            output_dir,
            axis_order="ZX",
            target_resolution=target_resolution,
            save_frames=save_frames,
            flip_z=True,
        )
        if frame_data:
            successful += 1
            # Only accept frames that have all required fields
            if all(k in frame_data for k in ("density", "velx", "velz")):
                d = frame_data["density"].astype(np.float32, copy=False)
                vx = frame_data["velx"].astype(np.float32, copy=False)
                vz = frame_data["velz"].astype(np.float32, copy=False)

                if hw_ref is None:
                    hw_ref = d.shape
                if d.shape != hw_ref or vx.shape != hw_ref or vz.shape != hw_ref:
                    print(
                        f"    Warning: skipping frame due to shape mismatch. Expected {hw_ref}, got d={d.shape}, vx={vx.shape}, vz={vz.shape}"
                    )
                else:
                    density_frames.append(d)
                    velx_frames.append(vx)
                    velz_frames.append(vz)

                # Check if we got actual data
                has_nonzero_data = np.any(d != 0) or np.any(vx != 0) or np.any(vz != 0)
                if has_nonzero_data:
                    total_nonzero_files += 1
            else:
                print(
                    "    Warning: missing one of required grids (density/velocity); frame will not be included in sequences."
                )

    # Report per-frame processing
    print("\n=== Per-frame extraction complete ===")
    print(f"Successfully processed: {successful}/{len(vdb_files)} files")
    print(f"Frames with non-zero data (accepted): {total_nonzero_files}")
    print(f"Accepted frames with all fields: {len(density_frames)}")

    # Pack into non-overlapping sequences and save npz
    N = len(density_frames)
    if N < seq_len:
        print(f"Not enough frames to create a single sequence (have {N}, need {seq_len}). No seq_*.npz written.")
        return

    T = seq_len
    H, W = hw_ref if hw_ref is not None else (target_resolution, target_resolution)
    n_seqs = N // seq_len
    print(f"Packing {N} frames into {n_seqs} non-overlapping sequences of length {seq_len}.")

    for s in range(n_seqs):
        start = s * seq_len
        end = start + seq_len
        d_stack = np.stack(density_frames[start:end], axis=0).astype(np.float32, copy=False)
        x_stack = np.stack(velx_frames[start:end], axis=0).astype(np.float32, copy=False)
        z_stack = np.stack(velz_frames[start:end], axis=0).astype(np.float32, copy=False)

        out_name = f"seq_{s:04d}.npz"
        out_path = output_dir / out_name
        np.savez_compressed(out_path, density=d_stack, velx=x_stack, velz=z_stack)
        print(f"Saved {out_path}  shape: T={T}, H={H}, W={W}")

    print("\n=== Complete ===")
    print(f"Generated {n_seqs} sequence files (seq_*.npz) in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VDB fluid cache directly into seq_*.npz sequences (sum density, avg velocity across Y)."
    )
    parser.add_argument("cache_dir", help="Directory containing VDB files")
    parser.add_argument("output_dir", help="Output directory for seq_*.npz files")
    parser.add_argument(
        "-r", "--resolution", type=int, default=32, help="Target resolution (square, e.g., 32 for 32x32)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save per-frame .npy files for debugging (density_####.npy, velx_####.npy, velz_####.npy)",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return

    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")

    process_all_frames(
        cache_dir=cache_dir,
        output_dir=output_dir,
        target_resolution=args.resolution,
        max_frames=args.max_frames,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()
