import numpy as np


def calculate_padding(
    min_x: int,
    max_x: int,
    min_z: int,
    max_z: int,
    target_resolution: int,
) -> tuple[int, int, int, int]:
    """
    Calculate padding needed to restore full domain coordinates.
    Computes padding amounts to expand the active voxel region to full domain size.
    """
    pad_x_before = max(0, min_x - 1)
    pad_x_after = max(0, target_resolution - max_x)
    pad_z_before = max(0, min_z - 1)
    pad_z_after = max(0, target_resolution - max_z)

    return pad_x_before, pad_x_after, pad_z_before, pad_z_after


def pad_to_domain_size(
    data: np.ndarray,
    pad_x_before: int,
    pad_x_after: int,
    pad_z_before: int,
    pad_z_after: int,
) -> np.ndarray:
    """
    Pad array to full domain size with zeros.
    """
    if pad_x_before > 0 or pad_x_after > 0 or pad_z_before > 0 or pad_z_after > 0:
        print(
            f"    Padding to domain size: X[{pad_x_before}, {pad_x_after}], Z[{pad_z_before}, {pad_z_after}]"
        )
        padded = np.pad(
            data,
            ((pad_z_before, pad_z_after), (pad_x_before, pad_x_after)),
            mode="constant",
            constant_values=0,
        )
        print(f"    Padded shape: {padded.shape}")
        return padded
    return data


def check_resolution_consistency(
    data: np.ndarray,
    target_resolution: int,
) -> None:
    assert data.shape[0] == target_resolution and data.shape[1] == target_resolution, (
        f"PADDING BUG: Expected shape ({target_resolution}, {target_resolution}), "
        f"got {data.shape}. Check padding calculation logic!"
    )


def apply_spatial_transforms(
    data: np.ndarray,
    axis_order: str,
    flip_x: bool,
    flip_z: bool,
    velocity_component: str | None = None,
) -> np.ndarray:
    """
    Apply axis reordering and flips to match desired coordinate system.

    Handles both scalar fields (density) and velocity components.
    When flipping axes, velocity components are negated for physical correctness.
    """
    # Apply axis reordering
    if axis_order.upper() == "XZ":
        result = np.swapaxes(data, 0, 1)  # (Z, X) -> (X, Z)
    elif axis_order.upper() == "ZX":
        result = data  # Keep (Z, X)
    else:
        raise ValueError("axis_order must be 'XZ' or 'ZX'")

    # Apply spatial flips
    # After axis reordering: if XZ then axis 0=X, axis 1=Z; if ZX then axis 0=Z, axis 1=X
    if flip_x:
        if axis_order.upper() == "XZ":
            result = np.flip(result, axis=0)
        else:  # ZX
            result = np.flip(result, axis=1)

        # Negate X-velocity component when flipping X axis for physical correctness
        if velocity_component == "velx":
            result = -result

    if flip_z:
        if axis_order.upper() == "XZ":
            result = np.flip(result, axis=1)
        else:  # ZX
            result = np.flip(result, axis=0)

        # Negate Z-velocity component when flipping Z axis for physical correctness
        if velocity_component == "velz":
            result = -result

    return result
