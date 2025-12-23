import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import vdb_config


@dataclass
class MeshTransform:
    translation: np.ndarray  # 3D vector [x, y, z]
    rotation: np.ndarray  # 3x3 rotation matrix
    scale: np.ndarray  # 3D vector [sx, sy, sz]

    def to_dict(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "scale": self.scale.tolist(),
        }


@dataclass
class MeshMetadata:
    name: str
    geometry_type: str
    vertex_count: int
    face_count: int
    transforms_per_frame: list[MeshTransform]  # Transform at each frame

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "geometry_type": self.geometry_type,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "transforms_per_frame": [t.to_dict() for t in self.transforms_per_frame],
        }


@dataclass
class AlembicMetadata:
    abc_path: Path
    cache_name: str
    frame_start: int
    frame_end: int
    meshes: list[MeshMetadata]

    def to_dict(self) -> dict:
        return {
            "abc_path": str(self.abc_path),
            "cache_name": self.cache_name,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "meshes": [mesh.to_dict() for mesh in self.meshes],
        }


def find_abc_for_cache(cache_dir: Path, blender_caches_root: Path) -> Path:
    cache_folder = cache_dir.parent
    cache_name = cache_folder.name

    abc_path = blender_caches_root / f"{cache_name}.abc"

    assert abc_path.exists(), f"No Alembic file found for cache: {cache_name} (expected: {abc_path})"
    assert abc_path.is_file(), f"No Alembic file found for cache: {cache_name} (expected: {abc_path})"

    return abc_path


def extract_abc_metadata(abc_path: Path, cache_name: str) -> AlembicMetadata:
    """
    Extract complete metadata from an Alembic file using Blender.
    """
    script_dir = Path(__file__).parent.parent / "blender_scripts"
    blender_script_path = script_dir / "extract_alembic_metadata.py"

    if not blender_script_path.exists():
        raise FileNotFoundError(f"Blender script not found: {blender_script_path}")

    try:
        blender_path = vdb_config.BLENDER_PATH
        if not blender_path or not blender_path.exists():
            raise Exception("Blender not found. Set BLENDER_PATH environment variable or ensure 'blender' is in PATH")

        blender_args = [str(blender_path), "--background", "--python", str(blender_script_path), "--", str(abc_path)]

        result = subprocess.run(blender_args, capture_output=True, text=True, timeout=60)

        output = result.stdout
        start_marker = "<<<METADATA_JSON_START>>>"
        end_marker = "<<<METADATA_JSON_END>>>"

        if start_marker in output and end_marker in output:
            start_idx = output.index(start_marker) + len(start_marker)
            end_idx = output.index(end_marker)
            json_str = output[start_idx:end_idx].strip()
            data = json.loads(json_str)
        else:
            raise Exception(f"Failed to extract metadata from Blender output. Stderr: {result.stderr}")

        meshes = []
        for mesh_data in data["meshes"]:
            transforms = []
            for t in mesh_data["transforms"]:
                transform = MeshTransform(
                    translation=np.array(t["translation"]), rotation=np.array(t["rotation"]), scale=np.array(t["scale"])
                )
                transforms.append(transform)

            mesh = MeshMetadata(
                name=mesh_data["name"],
                geometry_type=mesh_data["geometry_type"],
                vertex_count=mesh_data["vertex_count"],
                face_count=mesh_data["face_count"],
                transforms_per_frame=transforms,
            )
            meshes.append(mesh)

        return AlembicMetadata(
            abc_path=abc_path,
            cache_name=cache_name,
            frame_start=data["frame_start"],
            frame_end=data["frame_end"],
            meshes=meshes,
        )

    except Exception as e:
        raise Exception(f"Failed to extract Alembic metadata from {abc_path}: {e}") from e
