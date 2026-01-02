import numpy as np

from vdb_core.abc_metadata import AlembicMetadata, MeshMetadata, MeshTransform


def validate_emitter_meshes(abc_metadata: AlembicMetadata) -> list[MeshMetadata]:
    emitters = [m for m in abc_metadata.meshes if "Emitter" in m.name]

    if len(emitters) == 0:
        available_meshes = [m.name for m in abc_metadata.meshes]
        raise ValueError(
            f"No mesh containing 'Emitter' in name found in Alembic metadata. Available meshes: {available_meshes}"
        )

    for emitter in emitters:
        if emitter.geometry_type not in ["Cube", "Sphere"]:
            raise ValueError(
                f"Emitter '{emitter.name}' has unsupported geometry type '{emitter.geometry_type}'. Only 'Cube' and 'Sphere' are supported."
            )

    return emitters


def validate_collider_meshes(abc_metadata: AlembicMetadata) -> list[MeshMetadata] | None:
    colliders = [m for m in abc_metadata.meshes if "Collider" in m.name]

    if len(colliders) == 0:
        return None

    for collider in colliders:
        if collider.geometry_type not in ["Cube", "Sphere"]:
            raise ValueError(
                f"Collider '{collider.name}' has unsupported geometry type '{collider.geometry_type}'. Only 'Cube' and 'Sphere' are supported."
            )

    return colliders


def project_mesh_to_grid(
    transform: MeshTransform,
    geometry_type: str,
    grid_resolution: int,
) -> np.ndarray:
    pos_x_norm = transform.translation[0]  # X position in [-1, 1]
    pos_z_norm = transform.translation[2]  # Z position in [-1, 1]

    pos_x = (pos_x_norm + 1.0) * grid_resolution / 2.0
    pos_z = (pos_z_norm + 1.0) * grid_resolution / 2.0

    scale_x = transform.scale[0]
    scale_z = transform.scale[2]

    if scale_x <= 0 or scale_z <= 0:
        raise ValueError(f"Invalid scale: X={scale_x}, Z={scale_z}. Scale components must be positive.")

    scale_x_grid = scale_x * grid_resolution
    scale_z_grid = scale_z * grid_resolution

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
        mask = (np.abs(X_norm) <= 1.0) & (np.abs(Z_norm) <= 1.0)
    elif geometry_type == "Sphere":
        distance_sq = X_norm**2 + Z_norm**2
        mask = distance_sq <= 1.0
    else:
        raise ValueError(f"Unsupported geometry_type: {geometry_type}. Only 'Cube' and 'Sphere' are supported.")

    # Clip to exclude the outermost 1 cell on each side to create a border
    # Grid coordinates go from 0.5 to grid_resolution-0.5
    # Exclude cells where grid coordinate < 1.5 or > grid_resolution-1.5
    border_mask = (X >= 1.5) & (X <= grid_resolution - 1.5) & (Z >= 1.5) & (Z <= grid_resolution - 1.5)
    mask = mask & border_mask

    mesh_grid = mask.astype(np.float32)

    return mesh_grid
