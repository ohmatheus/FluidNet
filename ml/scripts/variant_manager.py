import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from copy import deepcopy


class VariantManager:
    def __init__(self, config_root: Path, checkpoints_dir: Path):
        self.config_root = config_root
        self.base_config_path = config_root / "base_config.yaml"
        self.architectures_dir = config_root / "model_architectures"
        self.variants_dir = config_root / "variants"
        self.checkpoints_dir = checkpoints_dir

        self._variants_cache = self.discover_variants()

    def discover_variants(self) -> Dict[str, Path]:
        """
        Recursively glob all .yaml files in variants directory.
        """
        variants = {}
        for yaml_file in self.variants_dir.rglob("*.yaml"):
            # Load YAML to get variant_name field
            try:
                data = self.load_yaml(yaml_file)
                variant_name = data.get("variant_name")
                if variant_name:
                    if variant_name in variants:
                        raise ValueError(
                            f"Duplicate variant name '{variant_name}' found:\n"
                            f"  {variants[variant_name]}\n"
                            f"  {yaml_file}"
                        )
                    variants[variant_name] = yaml_file
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
        return variants

    def get_variant_yaml_path(self, variant_name: str) -> Path:
        if variant_name not in self._variants_cache:
            raise FileNotFoundError(
                f"Variant '{variant_name}' not found in {self.variants_dir}\n"
                f"Available variants: {', '.join(sorted(self._variants_cache.keys()))}"
            )
        return self._variants_cache[variant_name]

    def get_variant_relative_dir(self, variant_name: str) -> Path:
        yaml_path = self.get_variant_yaml_path(variant_name)
        return yaml_path.parent.relative_to(self.variants_dir)

    def load_yaml(self, path: Path) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def deep_merge(self, base: Dict, override: Dict) -> Dict:
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def build_config_with_inheritance(self, variant_name: str) -> Dict:
        merged = self.load_yaml(self.base_config_path)

        # Load variant chain (from root to leaf)
        variant_chain = self.resolve_dependency_chain(variant_name)

        # Get architecture from first variant
        first_variant_path = self.get_variant_yaml_path(variant_chain[0])
        first_variant_data = self.load_yaml(first_variant_path)
        arch_name = first_variant_data.get("model_architecture")

        if not arch_name:
            raise ValueError(f"Variant {variant_chain[0]} missing 'model_architecture' field")

        # Merge
        arch_path = self.architectures_dir / f"{arch_name}.yaml"
        if not arch_path.exists():
            raise FileNotFoundError(f"Architecture '{arch_name}' not found at {arch_path}")

        arch_data = self.load_yaml(arch_path)
        merged = self.deep_merge(merged, arch_data)

        # Merge variant chain
        for variant in variant_chain:
            variant_path = self.get_variant_yaml_path(variant)
            variant_data = self.load_yaml(variant_path)

            # Validate architecture consistency (if specified)
            variant_arch = variant_data.get("model_architecture")
            if variant_arch is not None and variant_arch != arch_name:
                raise ValueError(
                    f"Architecture mismatch: {variant_chain[0]} uses {arch_name}, "
                    f"but {variant} uses {variant_arch}"
                )

            merged = self.deep_merge(merged, variant_data)

        merged["_variant_metadata"] = {
            "variant_name": variant_name,
            "model_architecture_name": arch_name,
            "full_model_name": f"{arch_name}_{variant_name}",
            "parent_variant": self.get_parent_variant(variant_name),
            "inheritance_chain": variant_chain,
            "relative_dir": self.get_variant_relative_dir(variant_name),
        }

        return merged

    def resolve_dependency_chain(self, variant_name: str) -> List[str]:
        chain = []
        current = variant_name
        visited: Set[str] = set()

        while current:
            if current in visited:
                raise ValueError(f"Circular dependency: {current} in chain {chain}")

            visited.add(current)
            chain.append(current)

            current_path = self.get_variant_yaml_path(current)
            variant_data = self.load_yaml(current_path)
            current = variant_data.get("parent_variant")

        return list(reversed(chain))  # Root to leaf

    def get_parent_variant(self, variant_name: str) -> Optional[str]:
        variant_path = self.get_variant_yaml_path(variant_name)
        variant_data = self.load_yaml(variant_path)
        return variant_data.get("parent_variant")

    def get_warmstart_checkpoint(self, variant_name: str) -> Optional[Path]:
        parent = self.get_parent_variant(variant_name)
        if not parent:
            return None

        parent_chain = self.resolve_dependency_chain(parent)
        arch_name = None
        for variant in parent_chain:
            variant_path = self.get_variant_yaml_path(variant)
            variant_data = self.load_yaml(variant_path)
            arch_name = variant_data.get("model_architecture")
            if arch_name:
                break

        if not arch_name:
            raise ValueError(f"Could not find model_architecture in chain for {parent}")

        parent_relative_dir = self.get_variant_relative_dir(parent)

        parent_full_name = f"{arch_name}_{parent}"
        checkpoint_path = self.checkpoints_dir / parent_relative_dir / parent_full_name / "best_model.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Parent checkpoint not found: {checkpoint_path}\n"
                f"Train {parent} before training {variant_name}"
            )

        return checkpoint_path

    def check_checkpoint_exists(self, variant_name: str) -> bool:
        variant_chain = self.resolve_dependency_chain(variant_name)
        arch_name = None
        for variant in variant_chain:
            variant_path = self.get_variant_yaml_path(variant)
            variant_data = self.load_yaml(variant_path)
            arch_name = variant_data.get("model_architecture")
            if arch_name:
                break

        if not arch_name:
            raise ValueError(f"Could not find model_architecture in chain for {variant_name}")

        relative_dir = self.get_variant_relative_dir(variant_name)

        full_name = f"{arch_name}_{variant_name}"
        checkpoint_path = self.checkpoints_dir / relative_dir / full_name / "best_model.pth"
        return checkpoint_path.exists()

    def topological_sort(self, variants: List[str]) -> List[str]:
        all_variants: Set[str] = set()
        for v in variants:
            all_variants.update(self.resolve_dependency_chain(v))

        # Build dependency graph
        graph: Dict[str, Set[str]] = {v: set() for v in all_variants}

        for v in all_variants:
            parent = self.get_parent_variant(v)
            if parent:
                graph[v].add(parent)

        in_degree = {v: len(deps) for v, deps in graph.items()}
        queue = [v for v in all_variants if in_degree[v] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for v, deps in graph.items():
                if node in deps:
                    deps.remove(node)
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(result) != len(all_variants):
            raise ValueError("Circular dependency detected")

        return result
