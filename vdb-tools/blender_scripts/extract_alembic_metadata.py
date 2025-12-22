import json
import sys

import bpy  # pyright: ignore


def extract_alembic_metadata(abc_path: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.alembic_import(filepath=abc_path)

    frame_start = int(bpy.context.scene.frame_start)
    frame_end = int(bpy.context.scene.frame_end)

    meshes_data = []

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            mesh = obj.data

            # Get vertex and face counts
            vertex_count = len(mesh.vertices)
            face_count = len(mesh.polygons)

            # Detect geometry type based on mesh characteristics
            geometry_type = "Unknown"
            if vertex_count == 8 and face_count in (6, 12):
                geometry_type = "Cube"
            elif vertex_count > 100 and face_count > 50:
                geometry_type = "Sphere"
            elif vertex_count < 100:
                geometry_type = "Mesh"
            else:
                geometry_type = "Mesh"

            # Extract transforms for each frame
            transforms = []
            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)

                matrix_world = obj.matrix_world.copy()
                translation, rotation, scale = matrix_world.decompose()
                # Convert rotation quaternion to matrix
                rotation_matrix = rotation.to_matrix()

                transform_data = {
                    "translation": [translation.x, translation.y, translation.z],
                    "rotation": [
                        [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2]],
                        [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2]],
                        [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]],
                    ],
                    "scale": [scale.x, scale.y, scale.z],
                }
                transforms.append(transform_data)

            mesh_data = {
                "name": obj.name,
                "geometry_type": geometry_type,
                "vertex_count": vertex_count,
                "face_count": face_count,
                "transforms": transforms,
            }
            meshes_data.append(mesh_data)

    output = {"frame_start": frame_start, "frame_end": frame_end, "meshes": meshes_data}

    return output


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        print(
            "Error: No arguments provided. Usage: blender --background --python extract_alembic_metadata.py -- <abc_path>"
        )
        sys.exit(1)

    if len(argv) < 1:
        print("Error: Alembic path required")
        sys.exit(1)

    abc_path = argv[0]

    metadata = extract_alembic_metadata(abc_path)

    print("<<<METADATA_JSON_START>>>")
    print(json.dumps(metadata))
    print("<<<METADATA_JSON_END>>>")
