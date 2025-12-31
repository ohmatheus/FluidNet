import json
import random
import sys
from pathlib import Path

import bpy  # pyright: ignore


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        raise ValueError("No arguments provided. Usage: blender --background --python create_random_simulation.py -- <json_params>")

    if len(argv) < 1:
        raise ValueError("JSON parameters required")

    params = json.loads(argv[0])
    required_keys = ["resolution", "frames", "cache_name", "output_dir", "blend_output_dir", "seed"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    return params


def create_fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_fluid_domain(resolution: int):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.scale = (1.0, 0.05, 1.0)

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain

    bpy.ops.object.modifier_add(type='FLUID')
    fluid_mod = domain.modifiers["Fluid"]
    fluid_mod.fluid_type = 'DOMAIN'

    domain_settings = fluid_mod.domain_settings
    domain_settings.domain_type = 'GAS'
    domain_settings.resolution_max = resolution

    domain_settings.use_collision_border_front = True
    domain_settings.use_collision_border_back = True
    domain_settings.use_collision_border_top = True
    domain_settings.use_collision_border_bottom = True
    domain_settings.use_collision_border_left = False
    domain_settings.use_collision_border_right = False

    domain_settings.vorticity = 0.1
    domain_settings.beta = 0.0
    domain_settings.delete_in_obstacle = True

    print(f"Created fluid domain: resolution={resolution}, vorticity=0.1, heat=0 (beta=0), delete_in_obstacle=True")
    return domain


def configure_emitter(obj):
    bpy.ops.object.modifier_add(type='FLUID')
    fluid_mod = obj.modifiers["Fluid"]
    fluid_mod.fluid_type = 'FLOW'

    flow_settings = fluid_mod.flow_settings
    flow_settings.flow_type = 'SMOKE'
    flow_settings.flow_behavior = 'INFLOW'
    flow_settings.density = 1.0
    flow_settings.temperature = 0.0
    flow_settings.flow_source = 'MESH'
    flow_settings.volume_density = 1.0
    flow_settings.surface_distance = 0.0


def configure_collider(obj):
    bpy.ops.object.modifier_add(type='FLUID')
    fluid_mod = obj.modifiers["Fluid"]
    fluid_mod.fluid_type = 'EFFECTOR'


def create_random_meshes(seed: int):
    random.seed(seed)

    num_meshes = random.randint(2, 6)
    print(f"Creating {num_meshes} random meshes (seed={seed})")

    meshes_info = []

    for i in range(num_meshes):
        mesh_type = random.choice(['CUBE', 'SPHERE'])
        position = (random.uniform(-1, 1), 0, random.uniform(-1, 1))

        if i == 0:
            role = 'Emitter'
        else:
            role = random.choice(['Emitter', 'Collider'])

        if mesh_type == 'CUBE':
            # Cubes get independent X and Z scales (Y stays 1.0 for projection)
            scale_x = random.uniform(0.1, 0.3)
            scale_z = random.uniform(0.1, 0.3)

            bpy.ops.mesh.primitive_cube_add(location=position)
            obj = bpy.context.active_object

            rotation_y = random.uniform(0, 360) * (3.14159 / 180)
            obj.rotation_euler = (0, rotation_y, 0)

            obj.scale = (scale_x, 0.1, scale_z)
        else:
            # Spheres use uniform scale
            uniform_scale = random.uniform(0.1, 0.3)

            bpy.ops.mesh.primitive_uv_sphere_add(location=position)
            obj = bpy.context.active_object

            obj.scale = (uniform_scale, uniform_scale, uniform_scale)

        obj.name = f"{role}_{i+1}"

        # Add animation data to match manually-created meshes
        obj.animation_data_create()

        if role == 'Emitter':
            configure_emitter(obj)
        else:
            configure_collider(obj)

        # Get actual scale from object (handles both cube and sphere cases)
        actual_scale = obj.scale

        meshes_info.append({
            "name": obj.name,
            "type": mesh_type,
            "role": role,
            "position": position,
            "scale": (actual_scale.x, actual_scale.y, actual_scale.z)
        })

        if mesh_type == 'CUBE':
            print(f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale=({actual_scale.x:.2f}, {actual_scale.y:.2f}, {actual_scale.z:.2f})")
        else:
            print(f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale={actual_scale.x:.2f}")

    return meshes_info


def bake_and_export(domain, params: dict, meshes_info: list):
    output_dir = Path(params["output_dir"])
    blend_output_dir = Path(params["blend_output_dir"])
    cache_name = params["cache_name"]
    frames = params["frames"]

    domain_settings = domain.modifiers["Fluid"].domain_settings
    domain_settings.cache_type = 'MODULAR'
    domain_settings.cache_data_format = 'OPENVDB'
    domain_settings.cache_directory = str(output_dir.absolute())

    domain_settings.cache_frame_start = 1
    domain_settings.cache_frame_end = frames

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    print(f"Baking simulation (frames 1-{frames})...")
    print(f"  Cache directory: {output_dir}")

    bpy.ops.object.select_all(action='DESELECT')
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain

    if domain_settings.has_cache_baked_data:
        print("  Freeing existing cache...")
        bpy.ops.fluid.free_all()

    bpy.ops.fluid.bake_all()

    data_dir = output_dir / "data"
    if not data_dir.exists():
        raise RuntimeError(f"Cache data directory not created: {data_dir}")

    vdb_files = list(data_dir.glob("*.vdb"))
    if len(vdb_files) == 0:
        raise RuntimeError("No VDB files generated")

    print(f"Bake complete: {len(vdb_files)} VDB files generated")

    bpy.ops.object.select_all(action='DESELECT')
    selected_count = 0
    for obj in bpy.data.objects:
        if 'Emitter' in obj.name or 'Collider' in obj.name:
            obj.select_set(True)
            selected_count += 1

    if selected_count == 0:
        raise RuntimeError("No emitter/collider meshes found for Alembic export")

    # Add imperceptible animation to trigger Alembic frame range export
    # Use the first emitter/collider mesh and add tiny position change
    first_mesh = None
    for obj in bpy.data.objects:
        if 'Emitter' in obj.name or 'Collider' in obj.name:
            first_mesh = obj
            break

    if first_mesh:
        # Add keyframe at frame 1 with original position
        bpy.context.scene.frame_set(1)
        first_mesh.keyframe_insert(data_path="location", frame=1)

        # Add keyframe at final frame with imperceptible offset (0.0001 units)
        bpy.context.scene.frame_set(frames)
        original_loc = first_mesh.location.copy()
        first_mesh.location.z += 0.0001  # Tiny offset that won't affect simulation
        first_mesh.keyframe_insert(data_path="location", frame=frames)
        first_mesh.location = original_loc  # Reset to original
        print(f"  Added imperceptible animation to {first_mesh.name} to trigger Alembic frame range")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    abc_path = output_dir.parent / f"{cache_name}.abc"
    print(f"Exporting Alembic (frames {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}): {abc_path}")

    bpy.ops.wm.alembic_export(
        filepath=str(abc_path),
        selected=True,
        start=1,
        end=frames,
    )

    blend_output_dir.mkdir(parents=True, exist_ok=True)
    blend_path = blend_output_dir / f"{cache_name}.blend"
    print(f"Saving blend file: {blend_path}")

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'WIREFRAME'

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(f"\n{'='*60}")
    print(f"Simulation complete: {cache_name}")
    print(f"  VDB files: {len(vdb_files)}")
    print(f"  Alembic: {abc_path.name}")
    print(f"  Blend file: {blend_path.name}")
    print(f"  Meshes: {len(meshes_info)} ({sum(1 for m in meshes_info if m['role'] == 'Emitter')} emitters, {sum(1 for m in meshes_info if m['role'] == 'Collider')} colliders)")
    print(f"{'='*60}")


def main():
    try:
        params = parse_args()

        print(f"\n{'='*60}")
        print(f"Creating simulation: {params['cache_name']}")
        print(f"  Resolution: {params['resolution']}")
        print(f"  Frames: {params['frames']}")
        print(f"  Seed: {params['seed']}")
        print(f"{'='*60}\n")

        create_fresh_scene()
        domain = create_fluid_domain(params["resolution"])
        meshes_info = create_random_meshes(params["seed"])
        bake_and_export(domain, params, meshes_info)

        print("\nSUCCESS\n")
        sys.exit(0)

    except Exception as e:
        print(f"\nX ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
