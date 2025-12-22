from pathlib import Path
from typing import Any

import openvdb  # type: ignore[import-not-found]


def extract_frame_number(vdb_path: Path) -> int:
    vdb_path = Path(vdb_path)
    basename = vdb_path.name
    try:
        parts = basename.replace(".vdb", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    except Exception:
        return 0


def get_grid_names(vdb_path: Path) -> list[str]:
    target_grids = ["density", "velocity"]  # add later here collision mask, emiter mask, etc
    available_grids: list[str] = []

    vdb_path_str = str(vdb_path)

    for grid_name in target_grids:
        try:
            _ = openvdb.read(vdb_path_str, grid_name)
            available_grids.append(grid_name)
        except Exception as e:
            print(f"  Warning: Could not read grid '{grid_name}' from {Path(vdb_path).name}: {e}")
            continue

    return available_grids


def get_grid_value(accessor: Any, x: int, y: int, z: int) -> Any:
    try:
        return accessor.getValue((x, y, z))
    except Exception:
        return accessor.getValue(x, y, z)
