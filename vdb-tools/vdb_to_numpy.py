import argparse
import shutil

from vdb_core.batch_processing import (
    aggregate_global_stats,
    compute_normalization_scales,
    process_all_cache_sequences,
    save_stats_to_yaml,
)

from config import PROJECT_ROOT_PATH, project_config, simulation_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert VDB fluid caches to seq_*.npz sequences for ML training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process per cache (for testing)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save per-frame .npy files for debugging (density_####.npy, velx_####.npy, velz_####.npy)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="all",
        help="Resolution to process (64, 128, 256, etc.) or 'all' to process all resolutions (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Specific split to process ('train', 'val', or 'test'). If not specified, processes all splits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing cache sequences (default: 1)",
    )

    args = parser.parse_args()

    blender_caches_root = (PROJECT_ROOT_PATH / project_config.vdb_tools.blender_cache_directory).resolve()
    npz_output_root = (PROJECT_ROOT_PATH / project_config.vdb_tools.npz_output_directory).resolve()

    if not blender_caches_root.exists():
        print(f"Error: Blender caches root directory not found: {blender_caches_root}")
        return

    if args.resolution == "all":
        resolutions = [d.name for d in blender_caches_root.iterdir() if d.is_dir() and d.name.isdigit()]
        if not resolutions:
            print(f"Error: No resolution subdirectories found in {blender_caches_root}")
            return
        print(f"Found resolutions: {', '.join(sorted(resolutions))}")
    else:
        resolutions = [args.resolution]

    if args.split is None:
        splits = simulation_config.splits.names
    else:
        splits = [args.split]

    for resolution_str in resolutions:
        resolution = int(resolution_str)

        # Collect stats from all splits for aggregation
        # NOTE: Stats computed across all splits (not just train) to ensure proper value
        # clamping to [0,1] or [-1,1]. Deviates from standard ML practice but required
        # for correct normalization across full dataset. 
        # The goal is autoregressive rollout, not generalization on unseen data.
        all_splits_stats: list = []
        all_splits_mesh_metadata: dict = {}

        for split_name in splits:
            cache_dir = blender_caches_root / resolution_str / split_name
            output_dir = npz_output_root / resolution_str / split_name

            if not cache_dir.exists():
                print(f"Warning: Split directory not found, skipping: {cache_dir}")
                continue

            print(f"\n{'=' * 70}")
            print(f"Processing: {resolution}x{resolution}, split: {split_name}")
            print(f"{'=' * 70}")
            print(f"Input: {cache_dir}")
            print(f"Output: {output_dir}")

            if output_dir.exists():
                print(f"Clearing output directory: {output_dir}")
                shutil.rmtree(output_dir)

            # Process split without saving stats (we'll aggregate all splits first)
            split_stats, split_mesh_metadata = process_all_cache_sequences(
                blender_caches_root=cache_dir,
                output_dir=output_dir,
                target_resolution=resolution,
                max_frames=args.max_frames,
                save_frames=args.save_frames,
                percentiles=project_config.vdb_tools.stats_percentiles,
                normalization_percentile=project_config.vdb_tools.normalization_percentile,
                stats_output_file=None,  # Don't save yet
                num_workers=args.workers,
            )

            all_splits_stats.extend(split_stats)
            all_splits_mesh_metadata.update(split_mesh_metadata)

        # Aggregate stats across all splits and save once
        if all_splits_stats:
            print(f"\n{'=' * 70}")
            print(f"Aggregating statistics across all splits for resolution {resolution}...")
            print(f"{'=' * 70}")

            global_stats = aggregate_global_stats(
                all_splits_stats, percentiles=project_config.vdb_tools.stats_percentiles
            )
            normalization_scales = compute_normalization_scales(
                global_stats, project_config.vdb_tools.normalization_percentile
            )
            stats_output_path = PROJECT_ROOT_PATH / project_config.vdb_tools.stats_output_file

            save_stats_to_yaml(
                output_path=stats_output_path,
                sequence_stats=all_splits_stats,
                global_stats=global_stats,
                normalization_scales=normalization_scales,
                mesh_metadata=all_splits_mesh_metadata,
            )

            print("\nSTATISTICS SUMMARY (aggregated across all splits):")
            print(f"  Total sequences analyzed: {global_stats.num_sequences}")
            print(f"  Density range: [{global_stats.density.min:.6f}, {global_stats.density.max:.6f}]")
            print(f"  Velx range: [{global_stats.velx.min:.6f}, {global_stats.velx.max:.6f}]")
            print(f"  Velz range: [{global_stats.velz.min:.6f}, {global_stats.velz.max:.6f}]")
            print(f"\n  Normalization scales (P{project_config.vdb_tools.normalization_percentile}):")
            print(f"    S_density = {normalization_scales.S_density:.6f}")
            print(f"    S_velx = {normalization_scales.S_velx:.6f}")
            print(f"    S_velz = {normalization_scales.S_velz:.6f}")
            print(f"\n  Saved aggregated statistics to: {stats_output_path}")
            print(f"{'=' * 70}")

    print(f"\n{'=' * 70}")
    print("All resolutions processed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
