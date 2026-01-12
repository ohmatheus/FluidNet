import argparse
import shutil

from vdb_core.batch_processing import process_all_cache_sequences

from config import PROJECT_ROOT_PATH, project_config


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

    for resolution_str in resolutions:
        resolution = int(resolution_str)
        cache_dir = blender_caches_root / resolution_str
        output_dir = npz_output_root / f"{resolution_str}"

        if not cache_dir.exists():
            print(f"Warning: Resolution directory not found, skipping: {cache_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Processing resolution: {resolution}x{resolution}")
        print(f"{'=' * 70}")
        print(f"Input directory: {cache_dir}")
        print(f"Output directory: {output_dir}")

        if output_dir.exists():
            print(f"Clearing output directory: {output_dir}")
            shutil.rmtree(output_dir)

        process_all_cache_sequences(
            blender_caches_root=cache_dir,
            output_dir=output_dir,
            target_resolution=resolution,
            max_frames=args.max_frames,
            save_frames=args.save_frames,
            percentiles=project_config.vdb_tools.stats_percentiles,
            normalization_percentile=project_config.vdb_tools.normalization_percentile,
            stats_output_file=project_config.vdb_tools.stats_output_file,
            num_workers=args.workers,
        )

    print(f"\n{'=' * 70}")
    print("All resolutions processed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
