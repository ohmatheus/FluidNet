import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from config import PROJECT_ROOT_PATH, project_config, simulation_config, vdb_config

BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts/create_random_simulation.py"


@dataclass
class SplitPlan:
    split_name: str
    total_count: int
    no_emitter_count: int
    no_collider_count: int
    collider_simple_count: int
    collider_medium_count: int
    collider_complex_count: int


def compute_split_plan(split_count: int, split_name: str, gen_config) -> SplitPlan:
    no_emitter_count = max(1, ceil(split_count * gen_config.distribution.no_emitter_pct))
    no_collider_count = max(1, ceil(no_emitter_count * gen_config.distribution.no_collider_pct))

    sims_with_emitters = split_count - no_emitter_count
    simple_thresh = gen_config.distribution.collider_mode_simple_threshold
    medium_thresh = gen_config.distribution.collider_mode_medium_threshold

    collider_simple = max(1, ceil(sims_with_emitters * simple_thresh))
    collider_medium = max(1, ceil(sims_with_emitters * (medium_thresh - simple_thresh)))
    collider_complex = max(0, sims_with_emitters - collider_simple - collider_medium)

    return SplitPlan(
        split_name=split_name,
        total_count=split_count,
        no_emitter_count=no_emitter_count,
        no_collider_count=no_collider_count,
        collider_simple_count=collider_simple,
        collider_medium_count=collider_medium,
        collider_complex_count=collider_complex,
    )


def assign_simulations_to_splits(total_sims: int, gen_config) -> dict[str, SplitPlan]:
    ratios = gen_config.splits.ratios
    names = gen_config.splits.names

    ratio_sum = sum(ratios)
    normalized_ratios = [r / ratio_sum for r in ratios]
    allocated = [int(r * total_sims) for r in normalized_ratios]

    leftover = total_sims - sum(allocated)
    if leftover > 0:
        fractional = [(r * total_sims) % 1 for r in normalized_ratios]
        sorted_idx = sorted(range(len(ratios)), key=lambda i: fractional[i], reverse=True)
        for i in range(leftover):
            allocated[sorted_idx[i]] += 1

    return {name: compute_split_plan(count, name, gen_config) for name, count in zip(names, allocated)}


def pack_config(gen_config, sim_index: int, base_seed: int, split_name: str, sim_type: dict) -> dict:
    em = gen_config.emitters
    col = gen_config.colliders

    return {
        "sim_index": sim_index,
        "split_name": split_name,
        "seed": base_seed + sim_index,
        **sim_type,
        "collider_mode": sim_type.get("collider_mode", "medium"),
        # Emitter config
        "emitter_count_range": list(em.count_range),
        "emitter_scale_min": em.scale.min,
        "emitter_scale_max_simple": em.scale.max_simple_mode,
        "emitter_scale_max": em.scale.max,
        "emitter_y_scale": em.scale.y_scale,
        "emitter_x_range": list(em.position.x_range),
        "emitter_z_range": list(em.position.z_range),
        "large_emitter_threshold": em.large_emitter.threshold,
        "large_emitter_x_range": list(em.large_emitter.x_range),
        "large_emitter_z": em.large_emitter.z_position,
        # Collider config
        "collider_count_medium_range": list(col.medium_mode.count_range),
        "collider_count_complex_range": list(col.complex_mode.count_range),
        "collider_simple_scale_min": col.simple_mode.scale.min,
        "collider_simple_scale_max": col.simple_mode.scale.max,
        "collider_simple_y_scale": col.simple_mode.scale.y_scale,
        "collider_complex_scale_min": col.complex_mode.scale.min,
        "collider_complex_scale_max": col.complex_mode.scale.max,
        "collider_z_range": list(col.position.z_range),
        # Domain & animation
        "domain_y_scale": gen_config.domain.y_scale,
        "domain_vorticity": gen_config.domain.vorticity,
        "domain_beta": gen_config.domain.beta,
        "anim_max_displacement": gen_config.animation.max_displacement,
    }


def generate_simulation_configs(
    split_plans: dict[str, SplitPlan], start_index: int, base_seed: int, gen_config
) -> list[tuple[int, str, dict]]:
    rng = random.Random(base_seed)
    all_configs = []
    sim_index = start_index

    for split_name, plan in split_plans.items():
        sim_types = []

        for i in range(plan.no_emitter_count):
            sim_types.append({
                "collider_mode": None,
                "no_emitters": True,
                "no_colliders": (i < plan.no_collider_count),
            })

        for _ in range(plan.collider_simple_count):
            sim_types.append({"collider_mode": "simple", "no_emitters": False, "no_colliders": False})
        for _ in range(plan.collider_medium_count):
            sim_types.append({"collider_mode": "medium", "no_emitters": False, "no_colliders": False})
        for _ in range(plan.collider_complex_count):
            sim_types.append({"collider_mode": "complex", "no_emitters": False, "no_colliders": False})

        rng.shuffle(sim_types)

        for sim_type in sim_types:
            config = pack_config(gen_config, sim_index, base_seed, split_name, sim_type)
            all_configs.append((sim_index, split_name, config))
            sim_index += 1

    return all_configs


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
    split_name: str,
    resolution: int,
    frames: int,
    output_base_dir: Path,
    blend_dir: Path,
    config_dict: dict,
) -> tuple[bool, str]:
    cache_name = f"cache_{sim_index:04d}"
    resolution_dir = output_base_dir / str(resolution) / split_name
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
        **config_dict,
        "resolution": resolution,
        "frames": frames,
        "cache_name": cache_name,
        "output_dir": str(cache_dir.absolute()),
        "blend_output_dir": str(blend_resolution_dir.absolute()),
    }

    blender_path = vdb_config.BLENDER_PATH
    if not blender_path or not blender_path.exists():
        raise FileNotFoundError(f"Blender not found at: {blender_path}. Set BLENDER_PATH in .env")

    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Blender script not found: {BLENDER_SCRIPT}")

    cmd = [str(blender_path), "--background", "--python", str(BLENDER_SCRIPT), "--", json.dumps(params)]

    print(f"\n[{split_name}] Simulation {sim_index}: {cache_name} (res={resolution}, frames={frames}, seed={config_dict['seed']})")

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
    sim_index, split_name, resolution, frames, output_base_dir, blend_dir, config_dict = task_args
    success, status = generate_simulation(
        sim_index, split_name, resolution, frames, output_base_dir, blend_dir, config_dict
    )
    return (sim_index, success, status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate batch of randomized Blender fluid simulations with split-based distribution",
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
    gen_config = simulation_config

    output_base_dir = PROJECT_ROOT_PATH / "data" / "blender_caches"
    blend_dir = PROJECT_ROOT_PATH / "data" / "simulations"

    split_plans = assign_simulations_to_splits(args.count, gen_config)

    print(f"\n{'=' * 70}")
    print("Split-Based Simulation Generation")
    print(f"{'=' * 70}")
    print("Configuration:")
    print(f"  Total count: {args.count}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Start index: {args.start_index}")
    print(f"  Frame range: [{args.min_frames}, {args.max_frames}]")
    print(f"  Base seed: {base_seed}")
    print(f"  Workers: {args.workers}")
    print(f"  Blender: {vdb_config.BLENDER_PATH}")
    print(f"\n{'=' * 70}")
    print("Split Plan:")
    print(f"{'=' * 70}")
    for split_name, plan in split_plans.items():
        print(f"\n{split_name.upper()} ({plan.total_count} sims):")
        print(f"  No emitters: {plan.no_emitter_count} (no colliders: {plan.no_collider_count})")
        print(f"  With emitters: {plan.total_count - plan.no_emitter_count}")
        print(f"    Simple: {plan.collider_simple_count}, Medium: {plan.collider_medium_count}, Complex: {plan.collider_complex_count}")
    print(f"{'=' * 70}\n")

    sim_configs = generate_simulation_configs(split_plans, args.start_index, base_seed, gen_config)

    for split_name in split_plans.keys():
        (output_base_dir / str(args.resolution) / split_name).mkdir(parents=True, exist_ok=True)

    tasks = []
    for sim_index, split_name, config_dict in sim_configs:
        frames = random.randint(args.min_frames, args.max_frames)
        tasks.append((sim_index, split_name, args.resolution, frames, output_base_dir, blend_dir, config_dict))

    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

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
