import json
import random
import sys
from pathlib import Path
from typing import Any

import bpy  # pyright: ignore


def parse_args() -> dict[str, Any]:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        raise ValueError(
            "No arguments provided. Usage: blender --background --python create_random_simulation.py -- <json_params>"
        )

    if len(argv) < 1:
        raise ValueError("JSON parameters required")

    params = json.loads(argv[0])
    required_keys = ["resolution", "frames", "cache_name", "output_dir", "blend_output_dir", "seed"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    return params


def create_fresh_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_fluid_domain(resolution: int) -> Any:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.scale = (1.0, 0.05, 1.0)

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain

    bpy.ops.object.modifier_add(type="FLUID")
    fluid_mod = domain.modifiers["Fluid"]
    fluid_mod.fluid_type = "DOMAIN"

    domain_settings = fluid_mod.domain_settings
    domain_settings.domain_type = "GAS"
    domain_settings.resolution_max = resolution

    domain_settings.use_collision_border_front = True
    domain_settings.use_collision_border_back = True
    domain_settings.use_collision_border_top = False
    domain_settings.use_collision_border_bottom = False
    domain_settings.use_collision_border_left = False
    domain_settings.use_collision_border_right = False

    domain_settings.vorticity = 0.05
    domain_settings.beta = 0.0
    domain_settings.delete_in_obstacle = True

    print(f"Created fluid domain: resolution={resolution}, vorticity=0.1, heat=0 (beta=0), delete_in_obstacle=True")
    return domain


def configure_emitter(obj: Any) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.modifier_add(type="FLUID")
    fluid_mod = obj.modifiers["Fluid"]
    fluid_mod.fluid_type = "FLOW"

    flow_settings = fluid_mod.flow_settings
    flow_settings.flow_type = "SMOKE"
    flow_settings.flow_behavior = "INFLOW"
    flow_settings.density = 1.0
    flow_settings.temperature = 0.0
    flow_settings.flow_source = "MESH"
    flow_settings.volume_density = 1.0
    flow_settings.surface_distance = 0.0


def configure_collider(obj: Any) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.modifier_add(type="FLUID")
    fluid_mod = obj.modifiers["Fluid"]
    fluid_mod.fluid_type = "EFFECTOR"


def clamp_to_bounds(position: tuple, margin: float = 0.1) -> tuple:
    min_val = -1.0 + margin
    max_val = 1.0 - margin

    x = max(min_val, min(max_val, position[0]))
    y = 0
    z = max(min_val, min(max_val, position[2]))

    return (x, y, z)


def animate_mesh(obj: Any, mesh_type: str, frames: int) -> None:
    start_pos = obj.location.copy()
    max_displacement = 1e-5

    middle_x = start_pos.x + random.uniform(-max_displacement, max_displacement)
    middle_z = start_pos.z + random.uniform(-max_displacement, max_displacement)
    middle_pos = (middle_x, 0, middle_z)

    end_x = start_pos.x + random.uniform(-max_displacement, max_displacement)
    end_z = start_pos.z + random.uniform(-max_displacement, max_displacement)
    end_pos = (end_x, 0, end_z)

    obj.location = start_pos
    obj.keyframe_insert(data_path="location", frame=1)

    obj.location = middle_pos
    obj.keyframe_insert(data_path="location", frame=frames // 2)

    obj.location = end_pos
    obj.keyframe_insert(data_path="location", frame=frames)

    for fcurve in obj.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"

    # if mesh_type == "CUBE":
    #    start_rot = obj.rotation_euler[1]

    #    rot_direction = random.choice([-1, 1])
    #    middle_rot = start_rot + rot_direction * random.uniform(1.57, 3.14)
    #    end_rot = middle_rot + rot_direction * random.uniform(1.57, 3.14)

    #    obj.rotation_euler[1] = start_rot
    #    obj.keyframe_insert(data_path="rotation_euler", index=1, frame=1)

    #    obj.rotation_euler[1] = middle_rot
    #    obj.keyframe_insert(data_path="rotation_euler", index=1, frame=frames // 2)

    #    obj.rotation_euler[1] = end_rot
    #    obj.keyframe_insert(data_path="rotation_euler", index=1, frame=frames)


def create_shape_component(position: tuple, scale: tuple, parent: Any, name: str) -> Any:
    bpy.ops.mesh.primitive_cube_add(location=position)
    cube = bpy.context.active_object
    cube.name = name
    cube.scale = scale

    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    cube.parent = parent

    return cube


def create_complex_shape(
    shape_type: str, center_position: tuple, role: str, overall_scale: float, rotation_y: float, index: int
) -> list[Any]:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=center_position)
    parent = bpy.context.active_object
    parent.name = f"{role}_{index}_Parent"
    parent.rotation_euler = (0, rotation_y, 0)

    cubes = []

    if shape_type == "T":
        cubes.append(create_shape_component((0, 0, 0), (0.1, 0.2, 0.6), parent, f"{role}_{index}_T_Vert"))
        cubes.append(create_shape_component((0, 0, 0.2), (0.6, 0.2, 0.1), parent, f"{role}_{index}_T_Horiz"))
    elif shape_type == "L":
        cubes.append(create_shape_component((0, 0, 0), (0.1, 0.2, 0.5), parent, f"{role}_{index}_L_Vert"))
        cubes.append(create_shape_component((0.2, 0, -0.2), (0.5, 0.2, 0.1), parent, f"{role}_{index}_L_Horiz"))
    elif shape_type == "cross":
        cubes.append(create_shape_component((0, 0, 0.3), (0.1, 0.2, 0.3), parent, f"{role}_{index}_Cross_N"))
        cubes.append(create_shape_component((0, 0, -0.3), (0.1, 0.2, 0.3), parent, f"{role}_{index}_Cross_S"))
        cubes.append(create_shape_component((0.3, 0, 0), (0.3, 0.2, 0.1), parent, f"{role}_{index}_Cross_E"))
        cubes.append(create_shape_component((-0.3, 0, 0), (0.3, 0.2, 0.1), parent, f"{role}_{index}_Cross_W"))
    elif shape_type == "stairs":
        cubes.append(create_shape_component((-0.4, 0, -0.3), (0.3, 0.2, 0.1), parent, f"{role}_{index}_Stair_1"))
        cubes.append(create_shape_component((-0.1, 0, -0.1), (0.3, 0.2, 0.1), parent, f"{role}_{index}_Stair_2"))
        cubes.append(create_shape_component((0.2, 0, 0.1), (0.3, 0.2, 0.1), parent, f"{role}_{index}_Stair_3"))

    parent.scale = (overall_scale, overall_scale, overall_scale)

    for cube in cubes:
        if role == "Emitter":
            configure_emitter(cube)
        else:
            configure_collider(cube)

    return [parent] + cubes


def create_random_meshes(
    seed: int, frames: int, collider_mode: str = "medium", no_emitters: bool = False, no_colliders: bool = False
) -> list[dict[str, Any]]:
    random.seed(seed)

    # Determine emitter and collider counts
    if no_emitters:
        num_emitters = 0
    else:
        num_emitters = random.randint(1, 2)

    if no_colliders:
        num_colliders = 0
    elif collider_mode == "simple":
        num_colliders = 0
    elif collider_mode == "medium":
        num_colliders = random.randint(1, 2)
    else:  # complex
        num_colliders = random.randint(2, 3)
    num_meshes = num_emitters + num_colliders

    print(
        f"Creating {num_meshes} random meshes ({num_emitters} emitters, {num_colliders} colliders, mode={collider_mode}, seed={seed})"
    )

    meshes_info = []

    roles = ["Emitter"] * num_emitters + ["Collider"] * num_colliders
    random.shuffle(roles)

    num_colliders = roles.count("Collider")
    num_simple_colliders = num_colliders // 2
    num_complex_colliders = num_colliders - num_simple_colliders

    collider_types = ["simple"] * num_simple_colliders + ["complex"] * num_complex_colliders
    random.shuffle(collider_types)
    collider_type_iter = iter(collider_types)

    # Alignment strategy for colliders: choose axis and base position
    align_axis = random.choice(["x", "z"])  # Which axis to align along
    if align_axis == "x":
        # Align colliders along X axis (varying X, fixed Z with small variation)
        align_base_z = random.uniform(0.1, 1.0)  # Upper region
    else:
        # Align colliders along Z axis (varying Z, fixed X with small variation)
        align_base_x = random.uniform(-0.5, 0.5)  # Center region

    collider_index = 0

    # First pass: place all emitters and record positions
    emitter_positions = []

    for i, role in enumerate(roles):
        if role == "Emitter":
            x = random.uniform(-1, 1)
            z = random.uniform(-1, -0.2)
            position = (x, 0, z)
            emitter_positions.append(position)
        else:
            if len(emitter_positions) > 0:
                emitter_center_x = sum(p[0] for p in emitter_positions) / len(emitter_positions)

                # Place colliders in upper region [0.1, 1.0] above emitters
                z = 0.1 + (collider_index * 0.9 / max(1, num_colliders - 1)) if num_colliders > 1 else 0.5
                x = emitter_center_x + random.uniform(-0.05, 0.05)
            else:
                if align_axis == "x":
                    x = -0.8 + (collider_index * 1.6 / max(1, num_colliders - 1)) if num_colliders > 1 else 0.0
                    x += random.uniform(-0.1, 0.1)
                    z = align_base_z + random.uniform(-0.05, 0.05)
                else:
                    z = 0.1 + (collider_index * 0.9 / max(1, num_colliders - 1)) if num_colliders > 1 else 0.5
                    z += random.uniform(-0.05, 0.05)
                    x = align_base_x + random.uniform(-0.05, 0.05)

            collider_index += 1
            position = (x, 0, z)

        # position = (x, 0, z)  # Moved up into if/else blocks

        if role == "Emitter":
            mesh_type = random.choice(["CUBE", "SPHERE"])

            if mesh_type == "CUBE":
                # Reduce max scale - was 0.3, now mode-dependent
                max_scale = 0.15 if collider_mode == "simple" else 0.2
                scale_x = random.uniform(0.1, max_scale)
                scale_z = random.uniform(0.1, max_scale)

                bpy.ops.mesh.primitive_cube_add(location=position)
                obj = bpy.context.active_object

                rotation_y = random.uniform(0, 360) * (3.14159 / 180)
                obj.rotation_euler = (0, rotation_y, 0)

                obj.scale = (scale_x, 0.1, scale_z)

                # For simple scenes with large emitters, center at bottom
                if collider_mode == "simple" and (scale_x > 0.12 or scale_z > 0.12):
                    pos_x = random.uniform(-0.6, 0.6)
                    position = (pos_x, 0, -0.75)
                    obj.location = position
            else:
                # Reduce max scale - was 0.3, now mode-dependent
                max_scale = 0.15 if collider_mode == "simple" else 0.2
                uniform_scale = random.uniform(0.1, max_scale)

                bpy.ops.mesh.primitive_uv_sphere_add(location=position)
                obj = bpy.context.active_object

                obj.scale = (uniform_scale, uniform_scale, uniform_scale)

                # For simple scenes with large emitters, center at bottom
                if collider_mode == "simple" and uniform_scale > 0.12:
                    pos_x = random.uniform(-0.6, 0.6)
                    position = (pos_x, 0, -0.75)
                    obj.location = position

            obj.name = f"{role}_{i + 1}"
            obj.animation_data_create()
            configure_emitter(obj)
            animate_mesh(obj, mesh_type, frames)

            actual_scale = obj.scale

            meshes_info.append(
                {
                    "name": obj.name,
                    "type": mesh_type,
                    "role": role,
                    "position": position,
                    "scale": (actual_scale.x, actual_scale.y, actual_scale.z),
                }
            )

            if mesh_type == "CUBE":
                print(
                    f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale=({actual_scale.x:.2f}, {actual_scale.y:.2f}, {actual_scale.z:.2f})"
                )
            else:
                print(
                    f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale={actual_scale.x:.2f}"
                )

        else:
            collider_type = next(collider_type_iter)

            if collider_type == "simple":
                mesh_type = random.choice(["CUBE", "SPHERE"])

                if mesh_type == "CUBE":
                    scale_x = random.uniform(0.08, 0.18)
                    scale_z = random.uniform(0.08, 0.18)

                    bpy.ops.mesh.primitive_cube_add(location=position)
                    obj = bpy.context.active_object

                    rotation_y = random.uniform(0, 360) * (3.14159 / 180)
                    obj.rotation_euler = (0, rotation_y, 0)

                    obj.scale = (scale_x, 0.1, scale_z)
                else:
                    uniform_scale = random.uniform(0.08, 0.18)

                    bpy.ops.mesh.primitive_uv_sphere_add(location=position)
                    obj = bpy.context.active_object

                    obj.scale = (uniform_scale, uniform_scale, uniform_scale)

                obj.name = f"{role}_{i + 1}"
                obj.animation_data_create()
                configure_collider(obj)

                if num_emitters == 0:
                    animate_mesh(obj, mesh_type, frames)

                actual_scale = obj.scale

                meshes_info.append(
                    {
                        "name": obj.name,
                        "type": mesh_type,
                        "role": role,
                        "position": position,
                        "scale": (actual_scale.x, actual_scale.y, actual_scale.z),
                    }
                )

                if mesh_type == "CUBE":
                    print(
                        f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale=({actual_scale.x:.2f}, {actual_scale.y:.2f}, {actual_scale.z:.2f})"
                    )
                else:
                    print(
                        f"  - {obj.name}: {mesh_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale={actual_scale.x:.2f}"
                    )

            else:
                shape_type = random.choice(["T", "L", "cross", "stairs"])
                overall_scale = random.uniform(0.3, 0.6)
                rotation_y = random.uniform(0, 360) * (3.14159 / 180)

                objects = create_complex_shape(shape_type, position, role, overall_scale, rotation_y, i + 1)

                parent = objects[0]
                parent.animation_data_create()

                if num_emitters == 0:
                    animate_mesh(parent, "CUBE", frames)

                meshes_info.append(
                    {
                        "name": parent.name,
                        "type": f"COMPLEX_{shape_type.upper()}",
                        "role": role,
                        "position": position,
                        "scale": overall_scale,
                        "num_components": len(objects) - 1,
                    }
                )

                print(
                    f"  - {parent.name}: COMPLEX_{shape_type.upper()} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale={overall_scale:.2f}, components={len(objects) - 1}"
                )

    return meshes_info


def bake_and_export(domain: Any, params: dict[str, Any], meshes_info: list[dict[str, Any]]) -> None:
    output_dir = Path(params["output_dir"])
    blend_output_dir = Path(params["blend_output_dir"])
    cache_name = params["cache_name"]
    frames = params["frames"]

    domain_settings = domain.modifiers["Fluid"].domain_settings
    domain_settings.cache_type = "MODULAR"
    domain_settings.cache_data_format = "OPENVDB"
    domain_settings.cache_directory = str(output_dir.absolute())

    domain_settings.cache_frame_start = 1
    domain_settings.cache_frame_end = frames

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    print(f"Baking simulation (frames 1-{frames})...")
    print(f"  Cache directory: {output_dir}")

    bpy.ops.object.select_all(action="DESELECT")
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

    bpy.ops.object.select_all(action="DESELECT")
    selected_count = 0
    for obj in bpy.data.objects:
        if "Emitter" in obj.name or "Collider" in obj.name:
            obj.select_set(True)
            selected_count += 1

    abc_path = output_dir.parent / f"{cache_name}.abc"

    if selected_count > 0:
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = frames

        print(f"Exporting Alembic (frames {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}): {abc_path}")

        bpy.ops.wm.alembic_export(
            filepath=str(abc_path),
            selected=True,
            start=1,
            end=frames,
        )
    else:
        print("Skipping Alembic export: no emitters or colliders in scene")

    blend_output_dir.mkdir(parents=True, exist_ok=True)
    blend_path = blend_output_dir / f"{cache_name}.blend"
    print(f"Saving blend file: {blend_path}")

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "WIREFRAME"

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(f"\n{'=' * 60}")
    print(f"Simulation complete: {cache_name}")
    print(f"  VDB files: {len(vdb_files)}")
    print(f"  Alembic: {abc_path.name}")
    print(f"  Blend file: {blend_path.name}")
    print(
        f"  Meshes: {len(meshes_info)} ({sum(1 for m in meshes_info if m['role'] == 'Emitter')} emitters, {sum(1 for m in meshes_info if m['role'] == 'Collider')} colliders)"
    )
    print(f"{'=' * 60}")


def main() -> None:
    try:
        params = parse_args()

        print(f"\n{'=' * 60}")
        print(f"Creating simulation: {params['cache_name']}")
        print(f"  Resolution: {params['resolution']}")
        print(f"  Frames: {params['frames']}")
        print(f"  Seed: {params['seed']}")
        print(f"{'=' * 60}\n")

        create_fresh_scene()
        domain = create_fluid_domain(params["resolution"])
        meshes_info = create_random_meshes(
            params["seed"],
            params["frames"],
            params.get("collider_mode", "medium"),
            params.get("no_emitters", False),
            params.get("no_colliders", False),
        )
        bake_and_export(domain, params, meshes_info)

        print("\nSUCCESS\n")
        sys.exit(0)

    except Exception as e:
        print(f"\nX ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
