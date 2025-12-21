import argparse
from pathlib import Path

from vdb_core.batch_processing import process_all_cache_sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert VDB fluid caches to seq_*.npz sequences for ML training."
    )
    parser.add_argument(
        "blender_caches_root", help="Root directory containing cache directories (e.g., ../data/Blender_caches/)"
    )
    parser.add_argument("output_dir", help="Output directory for seq_*.npz files")
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=32,
        help="Target resolution (square, e.g., 32 for 32x32)",
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

    blender_caches_root = Path(args.blender_caches_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not blender_caches_root.exists():
        print(f"Error: Blender caches root directory not found: {blender_caches_root}")
        return

    print(f"Blender caches root: {blender_caches_root}")
    print(f"Output directory: {output_dir}")
    print(f"Target resolution: {args.resolution}x{args.resolution}")

    process_all_cache_sequences(
        blender_caches_root=blender_caches_root,
        output_dir=output_dir,
        target_resolution=args.resolution,
        max_frames=args.max_frames,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    main()
