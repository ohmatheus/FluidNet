import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import vdb_config

PROJECT_ROOT = Path(__file__).parent.parent
BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts/create_random_simulation.py"

# todo move all of that to yaml config file
# and add explanation for each parameters
NO_EMITTER_PCT = 0.15
NO_COLLIDER_PCT = 0.50

COLLIDER_MODE_SIMPLE_THRESHOLD = 0.20
COLLIDER_MODE_MEDIUM_THRESHOLD = 0.80

EMITTER_COUNT_RANGE = (1, 2)
COLLIDER_COUNT_MEDIUM_RANGE = (1, 2)
COLLIDER_COUNT_COMPLEX_RANGE = (2, 3)

EMITTER_SCALE_MIN = 0.1
EMITTER_SCALE_MAX_SIMPLE = 0.2
EMITTER_SCALE_MAX = 0.3
EMITTER_Y_SCALE = 0.1

COLLIDER_SIMPLE_SCALE_MIN = 0.08
COLLIDER_SIMPLE_SCALE_MAX = 0.25
COLLIDER_SIMPLE_Y_SCALE = 0.1
COLLIDER_COMPLEX_SCALE_MIN = 0.3
COLLIDER_COMPLEX_SCALE_MAX = 0.8

EMITTER_X_RANGE = (-1.0, 1.0)
EMITTER_Z_RANGE = (-1.0, -0.2)
COLLIDER_Z_RANGE = (0.1, 1.0)
LARGE_EMITTER_THRESHOLD = 0.12
LARGE_EMITTER_X_RANGE = (-0.6, 0.6)
LARGE_EMITTER_Z = -0.75

DOMAIN_Y_SCALE = 0.05
DOMAIN_VORTICITY = 0.05
DOMAIN_BETA = 0.0

ANIM_MAX_DISPLACEMENT = 1e-5


def check_cache_exists(cache_dir: Path) -> bool:
    if not cache_dir.exists():
        return False

    data_dir = cache_dir / "data"
    if not data_dir.exists():
        return False

    vdb_files = list(data_dir.glob("*.vdb"))
    return len(vdb_files) > 0


def generate_simulation(
    sim_index: int,
    resolution: int,
    frames: int,
    output_base_dir: Path,
    blend_dir: Path,
    seed: int,
    collider_mode: str = "medium",
    no_emitters: bool = False,
    no_colliders: bool = False,
) -> tuple[bool, str]:
    cache_name = f"cache_{sim_index:04d}"
    resolution_dir = output_base_dir / str(resolution)
    cache_dir = resolution_dir / cache_name
    blend_resolution_dir = blend_dir / str(resolution)

    if check_cache_exists(cache_dir):
        return True, "skipped"

    if cache_dir.exists():
        print(f"  Warning: Incomplete cache found, deleting: {cache_dir}")
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    blend_resolution_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "resolution": resolution,
        "frames": frames,
        "cache_name": cache_name,
        "output_dir": str(cache_dir.absolute()),
        "blend_output_dir": str(blend_resolution_dir.absolute()),
        "seed": seed,
        "collider_mode": collider_mode,
        "no_emitters": no_emitters,
        "no_colliders": no_colliders,
        "emitter_count_range": list(EMITTER_COUNT_RANGE),
        "collider_count_medium_range": list(COLLIDER_COUNT_MEDIUM_RANGE),
        "collider_count_complex_range": list(COLLIDER_COUNT_COMPLEX_RANGE),
        "emitter_scale_min": EMITTER_SCALE_MIN,
        "emitter_scale_max_simple": EMITTER_SCALE_MAX_SIMPLE,
        "emitter_scale_max": EMITTER_SCALE_MAX,
        "emitter_y_scale": EMITTER_Y_SCALE,
        "collider_simple_scale_min": COLLIDER_SIMPLE_SCALE_MIN,
        "collider_simple_scale_max": COLLIDER_SIMPLE_SCALE_MAX,
        "collider_simple_y_scale": COLLIDER_SIMPLE_Y_SCALE,
        "collider_complex_scale_min": COLLIDER_COMPLEX_SCALE_MIN,
        "collider_complex_scale_max": COLLIDER_COMPLEX_SCALE_MAX,
        "emitter_x_range": list(EMITTER_X_RANGE),
        "emitter_z_range": list(EMITTER_Z_RANGE),
        "collider_z_range": list(COLLIDER_Z_RANGE),
        "large_emitter_threshold": LARGE_EMITTER_THRESHOLD,
        "large_emitter_x_range": list(LARGE_EMITTER_X_RANGE),
        "large_emitter_z": LARGE_EMITTER_Z,
        "domain_y_scale": DOMAIN_Y_SCALE,
        "domain_vorticity": DOMAIN_VORTICITY,
        "domain_beta": DOMAIN_BETA,
        "anim_max_displacement": ANIM_MAX_DISPLACEMENT,
    }

    blender_path = vdb_config.BLENDER_PATH
    if not blender_path or not blender_path.exists():
        raise FileNotFoundError(f"Blender not found at: {blender_path}. Set BLENDER_PATH in .env")

    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Blender script not found: {BLENDER_SCRIPT}")

    cmd = [str(blender_path), "--background", "--python", str(BLENDER_SCRIPT), "--", json.dumps(params)]

    print(f"\nSimulation {sim_index}: {cache_name} (res={resolution}, frames={frames}, seed={seed})")

    try:
        result = subprocess.run(cmd, timeout=3600)

        if result.returncode == 0:
            if check_cache_exists(cache_dir):
                return True, "success"
            else:
                print("  Error: Blender succeeded but no VDB files found")
                return False, "failed"
        else:
            print(f"  Error: Blender exited with code {result.returncode}")
            return False, "failed"

    except subprocess.TimeoutExpired:
        print("  Error: Simulation timed out after 1 hour")
        return False, "timeout"
    except Exception as e:
        print(f"  Error: {e}")
        return False, "error"


def worker_wrapper(task_args: tuple) -> tuple[int, bool, str]:
    sim_index, resolution, frames, output_base_dir, blend_dir, seed, collider_mode, no_emitters, no_colliders = (
        task_args
    )
    success, status = generate_simulation(
        sim_index, resolution, frames, output_base_dir, blend_dir, seed, collider_mode, no_emitters, no_colliders
    )
    return (sim_index, success, status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate batch of randomized Blender fluid simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--count", type=int, default=10, help="Number of simulations to generate (default: 10)")
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution (default: 128)")
    parser.add_argument("--start-index", type=int, default=1, help="Starting cache index (default: 1)")
    parser.add_argument("--min-frames", type=int, default=300, help="Minimum frame count (default: 300)")
    parser.add_argument("--max-frames", type=int, default=400, help="Maximum frame count (default: 400)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (default: current timestamp)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    if args.count <= 0:
        print("Error: --count must be positive")
        sys.exit(1)

    if args.resolution <= 0:
        print("Error: --resolution must be positive")
        sys.exit(1)

    if args.min_frames > args.max_frames:
        print("Error: --min-frames cannot be greater than --max-frames")
        sys.exit(1)

    if args.workers <= 0:
        print("Error: --workers must be positive")
        sys.exit(1)

    base_seed = args.seed if args.seed is not None else int(time.time())
    random.seed(base_seed)

    output_base_dir = PROJECT_ROOT / "data" / "blender_caches"
    blend_dir = PROJECT_ROOT / "data" / "simulations"

    print(f"\n{'=' * 70}")
    print("Blender Simulation Batch Generator")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Count: {args.count}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Start index: {args.start_index}")
    print(f"  Frame range: [{args.min_frames}, {args.max_frames}]")
    print(f"  Base seed: {base_seed}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_base_dir / str(args.resolution)}")
    print(f"  Blender: {vdb_config.BLENDER_PATH}")
    print(f"{'=' * 70}")

    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    tasks = []
    for i in range(args.count):
        sim_index = args.start_index + i
        frames = random.randint(args.min_frames, args.max_frames)
        sim_seed = base_seed + sim_index

        rand_val = random.random()
        if rand_val < COLLIDER_MODE_SIMPLE_THRESHOLD:
            collider_mode = "simple"
        elif rand_val < COLLIDER_MODE_MEDIUM_THRESHOLD:
            collider_mode = "medium"
        else:
            collider_mode = "complex"

        # Determine if simulation should have no emitters
        no_emitters = random.random() < NO_EMITTER_PCT
        no_colliders = False
        if no_emitters:
            # Of no-emitter sims, NO_COLLIDER_PCT have no colliders (empty)
            no_colliders = random.random() < NO_COLLIDER_PCT

        tasks.append(
            (
                sim_index,
                args.resolution,
                frames,
                output_base_dir,
                blend_dir,
                sim_seed,
                collider_mode,
                no_emitters,
                no_colliders,
            )
        )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(worker_wrapper, task): task for task in tasks}

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            sim_index, success, status = future.result()

            if status == "skipped":
                skipped += 1
                print(f"[{completed}/{args.count}] Simulation {sim_index}: Skipped")
            elif success:
                successful += 1
                print(f"[{completed}/{args.count}] Simulation {sim_index}: Success")
            else:
                failed += 1
                print(f"[{completed}/{args.count}] Simulation {sim_index}: Failed ({status})")

    elapsed_time = time.time() - start_time
    total_processed = successful + failed

    print(f"\n{'=' * 70}")
    print("Batch Generation Complete")
    print(f"{'=' * 70}")
    print("Results:")
    print(f"  Total requested: {args.count}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Total processed: {total_processed}")
    print("")
    print("Time:")
    print(f"  Total elapsed: {elapsed_time:.1f}s ({elapsed_time / 60:.1f}min)")
    if total_processed > 0:
        print(f"  Average per sim: {elapsed_time / total_processed:.1f}s")
    print(f"{'=' * 70}\n")

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
