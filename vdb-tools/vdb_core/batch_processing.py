import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import openvdb  # type: ignore[import-not-found]

from config import PROJECT_ROOT_PATH
from vdb_core.abc_metadata import AlembicMetadata, extract_abc_metadata, find_abc_for_cache
from vdb_core.grid_extraction import extract_density_field_avg, extract_velocity_components_avg
from vdb_core.mesh_projection import project_mesh_to_grid, validate_collider_meshes, validate_emitter_meshes
from vdb_core.statistics import (
    SequenceStats,
    aggregate_global_stats,
    compute_normalization_scales,
    compute_sequence_stats,
    save_stats_to_yaml,
)
from vdb_core.transforms import apply_spatial_transforms
from vdb_core.vdb_io import extract_frame_number, get_grid_names

AXIS_ORDER = "ZX"
FLIP_X = True
FLIP_Z = True


def process_vdb_file(
    vdb_path: Path,
    output_dir: Path,
    target_resolution: int = 32,
    save_frames: bool = False,
    axis_order: str = "XZ",
    flip_x: bool = True,
    flip_z: bool = False,
) -> dict[str, np.ndarray]:
    """
    Process single VDB file - avg density across y, weighthed (density) avg velocity across Y.
    """
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
                    density = extract_density_field_avg(grid, target_resolution)

                    # Apply spatial transformations
                    density = apply_spatial_transforms(
                        density,
                        axis_order=axis_order,
                        flip_x=flip_x,
                        flip_z=flip_z,
                        velocity_component=None,
                    )

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

                    # Apply spatial transformations to each component
                    for key in ("velx", "velz"):
                        if key in velocity_components:
                            arr = velocity_components[key]

                            # Apply transformations with velocity component tracking for negation
                            arr = apply_spatial_transforms(
                                arr,
                                axis_order=axis_order,
                                flip_x=flip_x,
                                flip_z=flip_z,
                                velocity_component=key,
                            )

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


def discover_cache_sequences(blender_caches_root: Path) -> list[Path]:
    cache_data_dirs = []

    if not blender_caches_root.exists():
        raise FileNotFoundError(f"Blender caches root not found: {blender_caches_root}")

    for cache_dir in sorted(blender_caches_root.iterdir()):
        if cache_dir.is_dir():
            data_dir = cache_dir / "data"
            if data_dir.exists() and data_dir.is_dir():
                cache_data_dirs.append(data_dir)

    return cache_data_dirs


def process_mesh_masks_for_frame(
    mesh_list: list | None,
    frame_idx: int,
    target_resolution: int,
    axis_order: str,
    flip_x: bool,
    flip_z: bool,
) -> np.ndarray:
    combined_mask = np.zeros((target_resolution, target_resolution), dtype=np.float32)

    if mesh_list is None:
        return combined_mask

    for mesh in mesh_list:
        transform = mesh.transforms_per_frame[frame_idx]
        single_mask = project_mesh_to_grid(
            transform=transform,
            geometry_type=mesh.geometry_type,
            grid_resolution=target_resolution,
        )

        single_mask = apply_spatial_transforms(
            single_mask,
            axis_order=axis_order,
            flip_x=flip_x,
            flip_z=flip_z,
            velocity_component=None,
        )

        # == logical OR
        combined_mask = np.maximum(combined_mask, single_mask)

    return combined_mask


def process_single_cache_sequence(
    cache_data_dir: Path,
    output_dir: Path,
    target_resolution: int,
    max_frames: int | None = None,
    save_frames: bool = False,
    starting_seq_number: int = 0,
    percentiles: list[int] | None = None,
    abc_metadata: AlembicMetadata | None = None,
) -> tuple[int, list[SequenceStats]]:
    cache_data_dir = Path(cache_data_dir)
    output_dir = Path(output_dir)

    meta_path = cache_data_dir.parent / "meta.json"
    cache_vorticity: float | None = None
    if meta_path.exists():
        with open(meta_path) as f:
            cache_vorticity = json.load(f).get("vorticity")
    else:
        print(f"Warning: meta.json missing for {cache_data_dir.parent.name}, vorticity will be null")

    vdb_files = sorted(cache_data_dir.glob("*.vdb"))

    if not vdb_files:
        print(f"No VDB files found in {cache_data_dir}")
        return 0, []

    print(f"Found {len(vdb_files)} VDB files")
    print(f"Target resolution: {target_resolution}x{target_resolution}")
    print("Processing: DENSITY (average across Y), VELOCITY (density-weighted average across Y)")

    # Deduce sequence length if not provided: use number of files that look like 'fluid_data_0001.vdb'
    pat = re.compile(r"^fluid_data_\d{4,}\.vdb$")
    candidate_files = [f for f in vdb_files if pat.match(f.name)]
    deduced = len(candidate_files) if candidate_files else len(vdb_files)
    seq_len = deduced
    print(f"Deduced sequence length: {seq_len} (from folder contents)")

    # Validate that ABC animation has same number of frames of the sequence length
    if abc_metadata:
        abc_frame_count = abc_metadata.frame_end - abc_metadata.frame_start + 1
        assert abc_frame_count == seq_len, (
            f"Alembic animation frame count mismatch: "
            f"ABC has {abc_frame_count} frames ({abc_metadata.frame_start}-{abc_metadata.frame_end}), "
            f"but sequence length is {seq_len}. "
            f"ABC animation must have at least {seq_len} frames to match the VDB sequence."
        )

    if max_frames:
        vdb_files = vdb_files[:max_frames]
        print(f"Processing first {len(vdb_files)} files")

    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_stats_list: list[SequenceStats] = []
    percentiles_to_use = percentiles if percentiles is not None else [75, 90, 95, 99, 100]

    successful = 0
    total_nonzero_files = 0

    density_frames = []  # List[np.ndarray]
    velx_frames = []
    velz_frames = []
    emitter_frames = []
    collider_frames = []
    hw_ref = None

    emitter_meshes = validate_emitter_meshes(abc_metadata)
    if emitter_meshes:
        print(
            f"Validated {len(emitter_meshes)} emitter mesh(es): {', '.join([f'{m.name} ({m.geometry_type})' for m in emitter_meshes])}"
        )
    else:
        print("No emitter meshes found")

    collider_meshes = validate_collider_meshes(abc_metadata)
    if collider_meshes is not None:
        print(
            f"Validated {len(collider_meshes)} collider mesh(es): {', '.join([f'{m.name} ({m.geometry_type})' for m in collider_meshes])}"
        )
    else:
        print("No collider meshes found")

    for frame_idx, vdb_file in enumerate(vdb_files):
        if abc_metadata and abc_metadata.meshes:
            # Frame number is 1-indexed in Alembic, frame_idx is 0-indexed in VDB processing
            abc_frame_idx = frame_idx  # Alembic frames start at frame_start
            if abc_frame_idx < len(abc_metadata.meshes[0].transforms_per_frame):
                print(f"  Mesh data (frame {abc_metadata.frame_start + frame_idx}):")
                for mesh in abc_metadata.meshes:
                    transform = mesh.transforms_per_frame[abc_frame_idx]
                    print(
                        f"    {mesh.name} ({mesh.geometry_type}): pos=[{transform.translation[0]:.3f}, {transform.translation[1]:.3f}, {transform.translation[2]:.3f}], scale=[{transform.scale[0]:.3f}, {transform.scale[1]:.3f}, {transform.scale[2]:.3f}]"
                    )

        frame_data = process_vdb_file(
            vdb_file,
            output_dir,
            axis_order=AXIS_ORDER,
            target_resolution=target_resolution,
            save_frames=save_frames,
            flip_x=FLIP_X,
            flip_z=FLIP_Z,
        )
        assert frame_data

        emitter_mask = process_mesh_masks_for_frame(
            mesh_list=emitter_meshes,
            frame_idx=frame_idx,
            target_resolution=target_resolution,
            axis_order=AXIS_ORDER,
            flip_x=FLIP_X,
            flip_z=FLIP_Z,
        )

        collider_mask = process_mesh_masks_for_frame(
            mesh_list=collider_meshes,
            frame_idx=frame_idx,
            target_resolution=target_resolution,
            axis_order=AXIS_ORDER,
            flip_x=FLIP_X,
            flip_z=FLIP_Z,
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

                if (
                    d.shape != hw_ref
                    or vx.shape != hw_ref
                    or vz.shape != hw_ref
                    or emitter_mask.shape != hw_ref
                    or collider_mask.shape != hw_ref
                ):
                    print(
                        f"    Warning: skipping frame due to shape mismatch. Expected {hw_ref}, got d={d.shape}, vx={vx.shape}, vz={vz.shape}, emitter={emitter_mask.shape}, collider={collider_mask.shape}"
                    )
                else:
                    density_frames.append(d)
                    velx_frames.append(vx)
                    velz_frames.append(vz)
                    emitter_frames.append(emitter_mask.astype(np.float32, copy=False))
                    collider_frames.append(collider_mask.astype(np.float32, copy=False))

                # Check if we got actual data
                has_nonzero_data = np.any(d != 0) or np.any(vx != 0) or np.any(vz != 0)
                if has_nonzero_data:
                    total_nonzero_files += 1
            else:
                raise AssertionError(
                    "Missing one of required grids (density/velocity); frame cannot be included in sequences."
                )

    print("\n=== Per-frame extraction complete ===")
    print(f"Successfully processed: {successful}/{len(vdb_files)} files")
    print(f"Frames with non-zero data (accepted): {total_nonzero_files}")
    print(f"Accepted frames with all fields: {len(density_frames)}")

    # Pack into non-overlapping sequences and save npz
    N = len(density_frames)
    if N < seq_len:
        print(f"Not enough frames to create a single sequence (have {N}, need {seq_len}). No seq_*.npz written.")
        return 0, []

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
        e_stack = np.stack(emitter_frames[start:end], axis=0).astype(np.float32, copy=False)
        c_stack = np.stack(collider_frames[start:end], axis=0).astype(np.float32, copy=False)

        # Use global sequence numbering
        global_seq_num = starting_seq_number + s
        out_name = f"seq_{global_seq_num:04d}.npz"
        out_path = output_dir / out_name
        np.savez_compressed(
            out_path,
            density=d_stack,
            velx=x_stack,
            velz=z_stack,
            emitter=e_stack,
            collider=c_stack,
        )
        print(f"Saved {out_path}  shape: T={T}, H={H}, W={W} (density, velx, velz, emitter, collider)")
        meta_out = output_dir / f"seq_{global_seq_num:04d}.meta.json"
        with open(meta_out, "w") as f:
            json.dump({"vorticity": cache_vorticity}, f)

        # Compute statistics for this sequence
        try:
            seq_stats = compute_sequence_stats(
                sequence_name=out_name.replace(".npz", ""),
                density=d_stack,
                velx=x_stack,
                velz=z_stack,
                percentiles=percentiles_to_use,
            )
            sequence_stats_list.append(seq_stats)
            print(f"  Computed statistics for {out_name}")
        except Exception as e:
            print(f"  Warning: Failed to compute statistics for {out_name}: {e}")

    print("\n=== Complete ===")
    print(f"Generated {n_seqs} sequence files in {output_dir}")

    return n_seqs, sequence_stats_list


def _process_cache_worker(
    args_tuple: tuple[Path, Path, Path, int, int | None, bool, list[int] | None, int],
) -> tuple[int, list[SequenceStats], str, dict | None]:
    (
        cache_data_dir,
        blender_caches_root,
        output_dir,
        target_resolution,
        max_frames,
        save_frames,
        percentiles,
        cache_number,
    ) = args_tuple

    cache_name = cache_data_dir.parent.name
    print(f"\n{'=' * 60}")
    print(f"Processing cache: {cache_name}")
    print(f"{'=' * 60}")

    abc_path = find_abc_for_cache(cache_data_dir, blender_caches_root)
    abc_metadata = extract_abc_metadata(abc_path, cache_name) if abc_path else None

    mesh_metadata = abc_metadata.to_dict() if abc_metadata else None

    sequences_from_cache, cache_stats = process_single_cache_sequence(
        cache_data_dir=cache_data_dir,
        output_dir=output_dir,
        target_resolution=target_resolution,
        max_frames=max_frames,
        save_frames=save_frames,
        starting_seq_number=cache_number,
        percentiles=percentiles,
        abc_metadata=abc_metadata,
    )

    print(f"Generated {sequences_from_cache} sequences from {cache_name}")

    return sequences_from_cache, cache_stats, cache_name, mesh_metadata


def process_all_cache_sequences(
    blender_caches_root: Path,
    output_dir: Path,
    target_resolution: int,
    max_frames: int | None = None,
    save_frames: bool = False,
    percentiles: list[int] | None = None,
    normalization_percentile: int = 95,
    stats_output_file: str | None = "data/_field_stats.yaml",
    num_workers: int = 1,
) -> tuple[list, dict]:
    cache_data_dirs = discover_cache_sequences(blender_caches_root)

    if not cache_data_dirs:
        print(f"No cache sequences found in {blender_caches_root}")
        return ([], {})

    print(f"Found {len(cache_data_dirs)} cache sequences to process:")
    for cache_data_dir in cache_data_dirs:
        print(f"  - {cache_data_dir.parent.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    percentiles_to_use = percentiles if percentiles is not None else [75, 90, 95, 99, 100]

    global_seq_counter = 0
    total_sequences = 0
    all_sequence_stats: list[SequenceStats] = []
    all_mesh_metadata = {}

    if num_workers == 1:
        for cache_data_dir in cache_data_dirs:
            cache_name = cache_data_dir.parent.name
            print(f"\n{'=' * 60}")
            print(f"Processing cache: {cache_name}")
            print(f"{'=' * 60}")

            abc_path = find_abc_for_cache(cache_data_dir, blender_caches_root)
            abc_metadata = extract_abc_metadata(abc_path, cache_name) if abc_path else None

            all_mesh_metadata[cache_name] = abc_metadata.to_dict() if abc_metadata else None

            sequences_from_cache, cache_stats = process_single_cache_sequence(
                cache_data_dir=cache_data_dir,
                output_dir=output_dir,
                target_resolution=target_resolution,
                max_frames=max_frames,
                save_frames=save_frames,
                starting_seq_number=global_seq_counter,
                percentiles=percentiles_to_use,
                abc_metadata=abc_metadata,
            )

            global_seq_counter += sequences_from_cache
            total_sequences += sequences_from_cache
            all_sequence_stats.extend(cache_stats)

            print(f"Generated {sequences_from_cache} sequences from {cache_name}")
    else:
        worker_args = []
        for cache_data_dir in cache_data_dirs:
            cache_name = cache_data_dir.parent.name
            cache_num_match = re.search(r"(\d+)", cache_name)
            cache_number = int(cache_num_match.group(1)) if cache_num_match else 0

            worker_args.append(
                (
                    cache_data_dir,
                    blender_caches_root,
                    output_dir,
                    target_resolution,
                    max_frames,
                    save_frames,
                    percentiles_to_use,
                    cache_number,
                )
            )

        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_cache_worker, worker_args)

        for sequences_from_cache, cache_stats, cache_name, mesh_metadata in results:
            all_mesh_metadata[cache_name] = mesh_metadata
            total_sequences += sequences_from_cache
            all_sequence_stats.extend(cache_stats)

    # Compute and optionally save global statistics
    if all_sequence_stats and stats_output_file is not None:
        try:
            global_stats = aggregate_global_stats(all_sequence_stats, percentiles=percentiles_to_use)
            normalization_scales = compute_normalization_scales(global_stats, normalization_percentile)
            stats_output_path = PROJECT_ROOT_PATH / stats_output_file

            save_stats_to_yaml(
                output_path=stats_output_path,
                sequence_stats=all_sequence_stats,
                global_stats=global_stats,
                normalization_scales=normalization_scales,
                mesh_metadata=all_mesh_metadata,
            )

            print(f"\n{'=' * 60}")
            print("STATISTICS SUMMARY:")
            print(f"  Total sequences analyzed: {global_stats.num_sequences}")
            print(f"  Density range: [{global_stats.density.min:.6f}, {global_stats.density.max:.6f}]")
            print(f"  Velx range: [{global_stats.velx.min:.6f}, {global_stats.velx.max:.6f}]")
            print(f"  Velz range: [{global_stats.velz.min:.6f}, {global_stats.velz.max:.6f}]")
            print(f"\n  Normalization scales (P{normalization_percentile}):")
            print(f"    S_density = {normalization_scales.S_density:.6f}")
            print(f"    S_velx = {normalization_scales.S_velx:.6f}")
            print(f"    S_velz = {normalization_scales.S_velz:.6f}")
            print(f"  Stats saved to: {stats_output_path}")
            print(f"{'=' * 60}")
        except Exception as e:
            print(f"\nError: Failed to save statistics: {e}")
            print("This is non-critical; NPZ files were created successfully.")
    elif not all_sequence_stats:
        print("\nWarning: No sequences were created, no statistics computed.")

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: Generated {total_sequences} total sequences in {output_dir}")
    print(f"{'=' * 60}")

    # Return stats and metadata for caller aggregation
    return (all_sequence_stats, all_mesh_metadata)
