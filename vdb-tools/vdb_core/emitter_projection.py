import numpy as np

from vdb_core.abc_metadata import AlembicMetadata, MeshMetadata, MeshTransform


def validate_emitter_mesh(abc_metadata: AlembicMetadata) -> MeshMetadata:
    emitters = [m for m in abc_metadata.meshes if m.name == "Emitter"]

    if len(emitters) == 0:
        available_meshes = [m.name for m in abc_metadata.meshes]
        raise ValueError(f"No mesh named 'Emitter' found in Alembic metadata. Available meshes: {available_meshes}")

    if len(emitters) > 1:
        raise ValueError(
            f"Multiple meshes named 'Emitter' found ({len(emitters)}). Only one emitter mesh is supported."
        )

    emitter = emitters[0]

    if emitter.geometry_type not in ["Cube", "Sphere"]:
        raise ValueError(
            f"Emitter geometry type '{emitter.geometry_type}' not supported. Only 'Cube' and 'Sphere' are supported."
        )

    return emitter


def project_mesh_to_grid(
    transform: MeshTransform,
    geometry_type: str,
    grid_resolution: int,
) -> np.ndarray:
    pos_x_norm = transform.translation[0]  # X position in [-1, 1]
    pos_z_norm = transform.translation[2]  # Z position in [-1, 1]

    # Convert to grid coordinates [0, grid_resolution]
    # grid_index = (position + 1) * grid_resolution / 2
    # Add 0.5 to account for cell-centered grid (cell centers are at i+0.5)
    pos_x = (pos_x_norm + 1.0) * grid_resolution / 2.0 - 1  # 0.5
    pos_z = (pos_z_norm + 1.0) * grid_resolution / 2.0 - 1  # 0.5

    scale_x = transform.scale[0]
    scale_z = transform.scale[2]

    if scale_x <= 0 or scale_z <= 0:
        raise ValueError(f"Invalid scale: X={scale_x}, Z={scale_z}. Scale components must be positive.")

    # Scale represents half-extent (radius) in Blender in normalized [-1,1] space
    # Convert to grid units: multiply by grid_resolution/2 to get radius in pixels
    scale_x_grid = scale_x * grid_resolution
    scale_z_grid = scale_z * grid_resolution

    # Grid coordinates - pixel centers
    x_coords = np.arange(grid_resolution, dtype=np.float32) + 0.5
    z_coords = np.arange(grid_resolution, dtype=np.float32) + 0.5

    Z, X = np.meshgrid(z_coords, x_coords, indexing="ij")

    X_local = X - pos_x
    Z_local = Z - pos_z

    # Apply rotation (Y-axis rotation affects XZ plane)
    # Extract rotation angle around Y axis from rotation matrix
    R = transform.rotation
    rotation_y = np.arctan2(R[2, 0], R[0, 0])  # Y-axis rotation angle

    # Apply inverse rotation to transform grid coordinates to object-local space
    cos_theta = np.cos(-rotation_y)
    sin_theta = np.sin(-rotation_y)
    X_rotated = cos_theta * X_local - sin_theta * Z_local
    Z_rotated = sin_theta * X_local + cos_theta * Z_local

    # Normalize by radius (scale_x_grid is already the radius in pixels)
    # After normalization, |coord| <= 1 means inside the shape
    X_norm = X_rotated / (scale_x_grid / 2.0)
    Z_norm = Z_rotated / (scale_z_grid / 2.0)

    if geometry_type == "Cube":
        # rectangle
        mask = (np.abs(X_norm) <= 1.0) & (np.abs(Z_norm) <= 1.0)
    elif geometry_type == "Sphere":
        # circle
        distance_sq = X_norm**2 + Z_norm**2
        mask = distance_sq <= 1.0
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}. Only 'Cube' and 'Sphere' are supported.")

    # Convert boolean mask to float32 - todo check later if binary can do
    mesh_grid = mask.astype(np.float32)

    return mesh_grid
