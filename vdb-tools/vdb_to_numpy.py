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

    args = parser.parse_args()

    blender_caches_root = (PROJECT_ROOT_PATH / project_config.vdb_tools.blender_cache_directory).resolve()
    output_dir = (PROJECT_ROOT_PATH / project_config.vdb_tools.npz_output_directory).resolve()

    if not blender_caches_root.exists():
        print(f"Error: Blender caches root directory not found: {blender_caches_root}")
        return

    resolution = project_config.simulation.grid_resolution

    print(f"Blender caches root: {blender_caches_root}")
    print(f"Output directory: {output_dir}")
    print(f"Target resolution: {resolution}x{resolution}")

    if output_dir.exists():
        print(f"Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    process_all_cache_sequences(
        blender_caches_root=blender_caches_root,
        output_dir=output_dir,
        target_resolution=resolution,
        max_frames=args.max_frames,
        save_frames=args.save_frames,
        percentiles=project_config.vdb_tools.stats_percentiles,
        normalization_percentile=project_config.vdb_tools.normalization_percentile,
        stats_output_file=project_config.vdb_tools.stats_output_file,
    )


if __name__ == "__main__":
    main()
