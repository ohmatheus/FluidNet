from dataclasses import dataclass
from typing import Any

import numpy as np

from vdb_core.transforms import calculate_padding, check_resolution_consistency, pad_to_domain_size
from vdb_core.vdb_io import get_grid_value


@dataclass
class BoundingBox:
    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int

    @property
    def dim_x(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def dim_y(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def dim_z(self) -> int:
        return self.max_z - self.min_z + 1


def get_bounding_box(grid: Any) -> BoundingBox | None:
    try:
        bbox_result = grid.evalActiveVoxelBoundingBox()

        if isinstance(bbox_result, tuple) and len(bbox_result) == 2:
            min_tuple, max_tuple = bbox_result
            min_x, min_y, min_z = min_tuple
            max_x, max_y, max_z = max_tuple

            # Check for empty/invalid bounds (min > max indicates no active voxels)
            if min_x > max_x or min_y > max_y or min_z > max_z:
                return None

            return BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            )
        return None
    except Exception:
        return None


def destagger_velocity_to_cell_centers(
    vel_x_data: np.ndarray, vel_z_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # vx is staggered in X (axis 1 of (Z,X) array): average adjacent columns
    padded_vx = np.pad(vel_x_data, ((0, 0), (0, 1)), mode="edge")
    vx_centered = (padded_vx[:, :-1] + padded_vx[:, 1:]) / 2

    # vz is staggered in Z (axis 0 of (Z,X) array): average adjacent rows
    padded_vz = np.pad(vel_z_data, ((0, 1), (0, 0)), mode="edge")
    vz_centered = (padded_vz[:-1, :] + padded_vz[1:, :]) / 2

    return vx_centered, vz_centered


def extract_density_field_avg(grid: Any, target_resolution: int) -> np.ndarray:
    """
    Extract density field by averaging across Y layers.

    Computes mean density per Y column, normalized by number of Y cells.
    Pads active voxel region to full domain (target_resolution x target_resolution) before resizing.
    """
    print(f"    Processing density grid (type: {type(grid)})")

    try:
        bbox = get_bounding_box(grid)
        if bbox is None:
            return np.zeros((target_resolution, target_resolution), dtype=np.float32)

        print(f"    Bounds: X[{bbox.min_x}:{bbox.max_x}], Y[{bbox.min_y}:{bbox.max_y}], Z[{bbox.min_z}:{bbox.max_z}]")
        print(f"    Averaging across {bbox.dim_y} Y layers, dimensions (X,Z): {bbox.dim_x}x{bbox.dim_z}")

        # Store as (Z, X) so rows=Z, cols=X
        density_sum = np.zeros((bbox.dim_z, bbox.dim_x), dtype=np.float32)
        accessor = grid.getAccessor()
        valid_samples = 0

        # Sum across all Y layers
        for y in range(bbox.min_y, bbox.max_y + 1):
            for i, x in enumerate(range(bbox.min_x, bbox.max_x + 1)):
                for j, z in enumerate(range(bbox.min_z, bbox.max_z + 1)):
                    try:
                        density_val = get_grid_value(accessor, x, y, z)
                        density_sum[j, i] += float(density_val)

                        if density_val > 0.0:
                            valid_samples += 1
                    except Exception:
                        continue

        # Compute average by dividing by number of Y layers
        print(f"    Valid density samples: {valid_samples}")
        print(f"    Density sum range (before averaging): [{density_sum.min():.6f}, {density_sum.max():.6f}]")

        density_data = density_sum / bbox.dim_y

        # Pad to full domain size
        pad_x_before, pad_x_after, pad_z_before, pad_z_after = calculate_padding(
            bbox.min_x, bbox.max_x, bbox.min_z, bbox.max_z, target_resolution
        )
        density_data = pad_to_domain_size(density_data, pad_x_before, pad_x_after, pad_z_before, pad_z_after)

        check_resolution_consistency(density_data, target_resolution)

        print(f"    Density range: [{density_data.min():.6f}, {density_data.max():.6f}]")

        return density_data

    except Exception as e:
        print(f"    Error extracting density: {e}")
        return np.zeros((target_resolution, target_resolution), dtype=np.float32)


def extract_velocity_components_avg(
    grid: Any, target_resolution: int, density_grid: Any = None
) -> dict[str, np.ndarray]:
    """
    Extract both X and Z velocity components by averaging across Y layers.

    If density_grid is provided, uses density-weighted averaging; otherwise uses uniform averaging.
    Pads active voxel region to full domain (target_resolution x target_resolution) before resizing.
    """
    print(f"    Processing velocity grid (type: {type(grid)})")

    try:
        bbox = get_bounding_box(grid)
        if bbox is None:
            return {
                "velx": np.zeros((target_resolution, target_resolution), dtype=np.float32),
                "velz": np.zeros((target_resolution, target_resolution), dtype=np.float32),
            }

        print(f"    Bounds: X[{bbox.min_x}:{bbox.max_x}], Y[{bbox.min_y}:{bbox.max_y}], Z[{bbox.min_z}:{bbox.max_z}]")
        print(f"    Averaging across {bbox.dim_y} Y layers, dimensions (X,Z): {bbox.dim_x}x{bbox.dim_z}")

        # IMPORTANT: store arrays as (Z, X) so that rows correspond to Z (vertical) and columns to X (horizontal)
        vel_x_sum = np.zeros((bbox.dim_z, bbox.dim_x), dtype=np.float32)
        vel_z_sum = np.zeros((bbox.dim_z, bbox.dim_x), dtype=np.float32)
        weight_sum = np.zeros((bbox.dim_z, bbox.dim_x), dtype=np.float32)

        accessor = grid.getAccessor()
        dens_accessor = density_grid.getAccessor() if density_grid is not None else None
        valid_samples = 0

        # Sum across all Y layers (density-weighted if density_grid provided)
        for y in range(bbox.min_y, bbox.max_y + 1):
            for i, x in enumerate(range(bbox.min_x, bbox.max_x + 1)):
                for j, z in enumerate(range(bbox.min_z, bbox.max_z + 1)):
                    try:
                        velocity_vec = get_grid_value(accessor, x, y, z)

                        # Extract X and Z components from 3D velocity vector
                        if hasattr(velocity_vec, "__len__") and len(velocity_vec) >= 3:
                            vx = float(velocity_vec[0])  # X component
                            vz = float(velocity_vec[2])  # Z component

                            # Get weight: density value if available, else 1.0
                            if dens_accessor is not None:
                                try:
                                    dens_val = get_grid_value(dens_accessor, x, y, z)
                                    weight = float(dens_val)
                                    if weight <= 0.0:
                                        continue  # Skip voxels with zero or negative density
                                except Exception:
                                    continue
                            else:
                                weight = 1.0  # Uniform averaging fallback

                            # Accumulate weighted sums
                            vel_x_sum[j, i] += weight * vx
                            vel_z_sum[j, i] += weight * vz
                            weight_sum[j, i] += weight

                            if vx != 0.0 or vz != 0.0:
                                valid_samples += 1

                    except Exception:
                        continue

        # Average by dividing by weight sum (avoid division by zero)
        vel_x_data = np.divide(vel_x_sum, weight_sum, out=np.zeros_like(vel_x_sum), where=weight_sum != 0)
        vel_z_data = np.divide(vel_z_sum, weight_sum, out=np.zeros_like(vel_z_sum), where=weight_sum != 0)

        print(f"    Valid velocity samples: {valid_samples}")

        # MAC staggered grid → cell-centered: average adjacent face values
        vel_x_data, vel_z_data = destagger_velocity_to_cell_centers(vel_x_data, vel_z_data)

        # Pad to full domain size
        pad_x_before, pad_x_after, pad_z_before, pad_z_after = calculate_padding(
            bbox.min_x, bbox.max_x, bbox.min_z, bbox.max_z, target_resolution
        )
        vel_x_data = pad_to_domain_size(vel_x_data, pad_x_before, pad_x_after, pad_z_before, pad_z_after)
        vel_z_data = pad_to_domain_size(vel_z_data, pad_x_before, pad_x_after, pad_z_before, pad_z_after)

        check_resolution_consistency(vel_x_data, target_resolution)
        check_resolution_consistency(vel_z_data, target_resolution)

        # Use keys compatible with the ML dataset (velx/velz)
        return {"velx": vel_x_data, "velz": vel_z_data}

    except Exception as e:
        print(f"    Error extracting velocity: {e}")
        return {
            "velx": np.zeros((target_resolution, target_resolution), dtype=np.float32),
            "velz": np.zeros((target_resolution, target_resolution), dtype=np.float32),
        }
