import openvdb
import numpy as np
import os
import glob
import argparse


def extract_frame_number(vdb_path):
    basename = os.path.basename(vdb_path)
    try:
        parts = basename.replace('.vdb', '').split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    except:
        return 0


def get_grid_names(vdb_path):
    target_grids = ['density', 'velocity']
    available_grids = []

    for grid_name in target_grids:
        try:
            grid = openvdb.read(vdb_path, grid_name)
            available_grids.append(grid_name)
        except:
            continue

    return available_grids


def extract_velocity_components_avg(grid, target_resolution=32):
    """
    Extract both X and Z velocity components by averaging across Y layers
    """
    print(f"    Processing velocity grid (type: {type(grid)})")

    try:
        bbox_result = grid.evalActiveVoxelBoundingBox()

        if isinstance(bbox_result, tuple) and len(bbox_result) == 2:
            min_tuple, max_tuple = bbox_result
            min_x, min_y, min_z = min_tuple
            max_x, max_y, max_z = max_tuple

            print(f"    Bounds: X[{min_x}:{max_x}], Y[{min_y}:{max_y}], Z[{min_z}:{max_z}]")

            dim_x = max_x - min_x + 1
            dim_y = max_y - min_y + 1
            dim_z = max_z - min_z + 1

            print(f"    Averaging across {dim_y} Y layers, dimensions: {dim_x}x{dim_z}")

            # Create output arrays for both X and Z components
            vel_x_sum = np.zeros((dim_x, dim_z), dtype=np.float32)
            vel_z_sum = np.zeros((dim_x, dim_z), dtype=np.float32)
            count_array = np.zeros((dim_x, dim_z), dtype=np.int32)

            accessor = grid.getAccessor()
            valid_samples = 0

            # Sum across all Y layers
            for y in range(min_y, max_y + 1):
                for i, x in enumerate(range(min_x, max_x + 1)):
                    for j, z in enumerate(range(min_z, max_z + 1)):
                        try:
                            # Try different coordinate formats
                            try:
                                velocity_vec = accessor.getValue((x, y, z))
                            except:
                                try:
                                    velocity_vec = accessor.getValue(x, y, z)
                                except:
                                    continue

                            # Extract X and Z components from 3D velocity vector
                            if hasattr(velocity_vec, '__len__') and len(velocity_vec) >= 3:
                                vel_x_sum[i, j] += float(velocity_vec[0])  # X component
                                vel_z_sum[i, j] += float(velocity_vec[2])  # Z component
                                count_array[i, j] += 1

                                if velocity_vec[0] != 0.0 or velocity_vec[2] != 0.0:
                                    valid_samples += 1

                        except:
                            continue

            # Average by dividing by count (avoid division by zero)
            vel_x_data = np.divide(vel_x_sum, count_array, out=np.zeros_like(vel_x_sum), where=count_array!=0)
            vel_z_data = np.divide(vel_z_sum, count_array, out=np.zeros_like(vel_z_sum), where=count_array!=0)

            print(f"    Valid velocity samples: {valid_samples}")

            # Resize to target resolution
            if (vel_x_data.shape[0] != target_resolution or vel_x_data.shape[1] != target_resolution):
                from scipy.ndimage import zoom
                zoom_x = target_resolution / vel_x_data.shape[0]
                zoom_z = target_resolution / vel_x_data.shape[1]

                vel_x_data = zoom(vel_x_data, (zoom_x, zoom_z), order=1)
                vel_z_data = zoom(vel_z_data, (zoom_x, zoom_z), order=1)

            print(f"    Vel X range: [{vel_x_data.min():.6f}, {vel_x_data.max():.6f}]")
            print(f"    Vel Z range: [{vel_z_data.min():.6f}, {vel_z_data.max():.6f}]")

            return {'vel_x': vel_x_data, 'vel_z': vel_z_data}

        else:
            return {'vel_x': np.zeros((target_resolution, target_resolution), dtype=np.float32),
                    'vel_z': np.zeros((target_resolution, target_resolution), dtype=np.float32)}

    except Exception as e:
        print(f"    Error extracting velocity: {e}")
        return {'vel_x': np.zeros((target_resolution, target_resolution), dtype=np.float32),
                'vel_z': np.zeros((target_resolution, target_resolution), dtype=np.float32)}


def extract_density_field_sum(grid, target_resolution=32):
    """
    Extract density field by summing across Y layers
    """
    print(f"    Processing density grid (type: {type(grid)})")

    try:
        bbox_result = grid.evalActiveVoxelBoundingBox()

        if isinstance(bbox_result, tuple) and len(bbox_result) == 2:
            min_tuple, max_tuple = bbox_result
            min_x, min_y, min_z = min_tuple
            max_x, max_y, max_z = max_tuple

            print(f"    Bounds: X[{min_x}:{max_x}], Y[{min_y}:{max_y}], Z[{min_z}:{max_z}]")

            dim_x = max_x - min_x + 1
            dim_y = max_y - min_y + 1
            dim_z = max_z - min_z + 1

            print(f"    Summing across {dim_y} Y layers, dimensions: {dim_x}x{dim_z}")

            density_data = np.zeros((dim_x, dim_z), dtype=np.float32)
            accessor = grid.getAccessor()
            valid_samples = 0

            # Sum across all Y layers
            for y in range(min_y, max_y + 1):
                for i, x in enumerate(range(min_x, max_x + 1)):
                    for j, z in enumerate(range(min_z, max_z + 1)):
                        try:
                            # Try different coordinate formats
                            try:
                                density_val = accessor.getValue((x, y, z))
                            except:
                                try:
                                    density_val = accessor.getValue(x, y, z)
                                except:
                                    continue

                            density_data[i, j] += float(density_val)

                            if density_val > 0.0:
                                valid_samples += 1

                        except:
                            continue

            print(f"    Valid density samples: {valid_samples}")

            # Resize to target resolution
            if (density_data.shape[0] != target_resolution or density_data.shape[1] != target_resolution):
                from scipy.ndimage import zoom
                zoom_x = target_resolution / density_data.shape[0]
                zoom_z = target_resolution / density_data.shape[1]
                density_data = zoom(density_data, (zoom_x, zoom_z), order=1)

            print(f"    Density range: [{density_data.min():.6f}, {density_data.max():.6f}]")

            return density_data

        else:
            return np.zeros((target_resolution, target_resolution), dtype=np.float32)

    except Exception as e:
        print(f"    Error extracting density: {e}")
        return np.zeros((target_resolution, target_resolution), dtype=np.float32)


def process_vdb_file(vdb_path, output_dir, target_resolution=32):
    """Process single VDB file - sum density, average velocity across Y"""

    print(f"\nProcessing: {os.path.basename(vdb_path)}")

    try:
        available_grids = get_grid_names(vdb_path)

        if not available_grids:
            print(f"  No density/velocity grids found")
            return {}

        print(f"  Available grids: {available_grids}")

        frame_num = extract_frame_number(vdb_path)
        frame_data = {}

        for grid_name in available_grids:
            print(f"  Processing '{grid_name}':")

            try:
                grid = openvdb.read(vdb_path, grid_name)

                if grid_name == 'density':
                    density = extract_density_field_sum(grid, target_resolution)
                    frame_data['density'] = density

                    # Save density
                    output_path = os.path.join(output_dir, f"density_{frame_num:04d}.npy")
                    np.save(output_path, density)
                    print(f"    Saved: density_{frame_num:04d}.npy")

                elif grid_name == 'velocity':
                    velocity_components = extract_velocity_components_avg(grid, target_resolution)
                    frame_data.update(velocity_components)

                    # Save velocity components
                    for comp_name, comp_data in velocity_components.items():
                        output_path = os.path.join(output_dir, f"{comp_name}_{frame_num:04d}.npy")
                        np.save(output_path, comp_data)
                        print(f"    Saved: {comp_name}_{frame_num:04d}.npy")

            except Exception as e:
                print(f"    Error processing '{grid_name}': {e}")

        return frame_data

    except Exception as e:
        print(f"  Error: {e}")
        return {}


def process_all_frames(cache_dir, output_dir, target_resolution=32, max_frames=None):
    """Process all VDB files in directory"""

    vdb_files = sorted(glob.glob(os.path.join(cache_dir, "*.vdb")))

    if not vdb_files:
        print(f"No VDB files found in {cache_dir}")
        return

    print(f"Found {len(vdb_files)} VDB files")
    print(f"Target resolution: {target_resolution}x{target_resolution}")
    print(f"Processing: DENSITY (sum across Y), VELOCITY (average across Y)")

    if max_frames:
        vdb_files = vdb_files[:max_frames]
        print(f"Processing first {len(vdb_files)} files")

    os.makedirs(output_dir, exist_ok=True)

    # Install scipy if needed
    try:
        from scipy.ndimage import zoom
    except ImportError:
        print("Installing scipy...")
        os.system("pip install scipy")

    # Process files
    successful = 0
    total_nonzero_files = 0

    for i, vdb_file in enumerate(vdb_files):
        frame_data = process_vdb_file(vdb_file, output_dir, target_resolution=target_resolution)
        if frame_data:
            successful += 1
            # Check if we got actual data
            has_nonzero_data = any(np.any(data != 0) for data in frame_data.values())
            if has_nonzero_data:
                total_nonzero_files += 1

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i + 1}/{len(vdb_files)} files processed")

    print(f"\n=== Complete ===")
    print(f"Successfully processed: {successful}/{len(vdb_files)} files")
    print(f"Files with non-zero data: {total_nonzero_files}")

    # Show results
    output_files = sorted(glob.glob(os.path.join(output_dir, "*.npy")))
    print(f"Generated {len(output_files)} numpy files")

    # Group by type and show summary
    density_files = [f for f in output_files if 'density' in f]
    vel_x_files = [f for f in output_files if 'vel_x' in f]
    vel_z_files = [f for f in output_files if 'vel_z' in f]

    print(f"  density: {len(density_files)} files")
    print(f"  vel_x: {len(vel_x_files)} files")
    print(f"  vel_z: {len(vel_z_files)} files")

    # Sample some files to verify
    print("\nSample verification:")
    for file_type, files in [('density', density_files[:3]), ('vel_x', vel_x_files[:3]), ('vel_z', vel_z_files[:3])]:
        for f in files:
            data = np.load(f)
            basename = os.path.basename(f)
            nonzero_count = np.count_nonzero(data)
            print(f"  {basename}: {data.shape}, range=[{data.min():.6f}, {data.max():.6f}], nonzero={nonzero_count}")


def main():
    parser = argparse.ArgumentParser(description='Convert VDB fluid cache to NumPy arrays (sum density, avg velocity)')
    parser.add_argument('cache_dir', help='Directory containing VDB files')
    parser.add_argument('output_dir', help='Output directory for NumPy files')
    parser.add_argument('-r', '--resolution', type=int, default=32, 
                        help='Target resolution (square, e.g., 32 for 32x32)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')

    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(cache_dir):
        print(f"Error: Cache directory not found: {cache_dir}")
        return

    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")

    process_all_frames(
        cache_dir=cache_dir,
        output_dir=output_dir,
        target_resolution=args.resolution,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # Default mode
        cache_dir = "/home/ohmatheus/Projects/Blender/cache2/data/"
        output_dir = "./numpy_output"
        target_resolution = 32

        print("No arguments provided, using defaults:")
        print(f"  Cache: {cache_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Resolution: {target_resolution}x{target_resolution}")
        print("  Method: SUM density, AVERAGE velocity across Y layers")

        if os.path.exists(cache_dir):
            process_all_frames(cache_dir, output_dir, target_resolution=target_resolution)
        else:
            print(f"Error: Default cache directory not found: {cache_dir}")
            print("Usage: python vdb_to_numpy_updated.py <cache_dir> <output_dir> [-r resolution]")
    else:
        main()