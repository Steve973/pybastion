"""
Configuration loader for the pybastion integration analysis pipeline.

The current pipeline is inventory-first:

  Stage 1: Build the EI-level call graph from inventory files
  Stage 2: Trace feature-flow cases from feature markers and the EI CFG
  Stage 3: Generate integration test specifications
  Stage 4: Split integration test specs

Path configuration uses a parent-relative model rather than placeholder
substitution. Absolute paths are used as-is. Relative child paths are resolved
against their owning parent path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pybastion_common.common_config import (
    get_bool,
    get_int,
    get_str,
    load_toml_config,
    require_table,
    select_config_table,
)


class StoreInConfig(argparse.Action):
    def __init__(self, option_strings, dest, config_obj, setter_method, **kwargs):
        self.config_obj = config_obj
        self.setter_method = setter_method
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        method = getattr(self.config_obj, self.setter_method)
        method(values)


_MODULE_DIR = Path(__file__).parent

_TARGET_ROOT: Path | None = None
_CONFIG: dict[str, Any] = {}

_PATH_OVERRIDES: dict[str, Path | str] = {}
_STAGE_OVERRIDES: dict[str, str] = {}
_GRAPH_CHECKER_OVERRIDES: dict[str, Path | str | bool] = {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return {}

    path = config_path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    payload = load_toml_config(path)
    return select_config_table("integration", payload)


def configure(
    *,
    config_path: Path | None = None,
    target_root: Path | str | None = None,
) -> None:
    global _CONFIG

    _CONFIG = load_config(config_path)
    set_target_root(target_root)


def get_config() -> dict[str, Any]:
    return _CONFIG


def set_target_root(path: Path | str | None) -> None:
    global _TARGET_ROOT

    _TARGET_ROOT = Path(path).resolve() if path else None


def get_target_root() -> Path:
    return _TARGET_ROOT or Path.cwd()


def set_path_override(name: str, value: Path | str | None) -> None:
    if value is not None:
        _PATH_OVERRIDES[name] = value


def set_stage_override(name: str, value: str | None) -> None:
    if value is not None:
        _STAGE_OVERRIDES[name] = value


def set_graph_checker_override(name: str, value: Path | str | bool | None) -> None:
    if value is not None:
        _GRAPH_CHECKER_OVERRIDES[name] = value


def resolve_path(
    config_value: str | Path,
    *,
    base: Path | None = None,
    relative_to_target: bool = True,
    allow_absolute: bool = True,
) -> Path:
    path = Path(config_value)

    if allow_absolute and path.is_absolute():
        return path

    if base is not None:
        return base / path

    if relative_to_target:
        return get_target_root() / path

    return _MODULE_DIR / path


def _section(name: str) -> dict[str, Any]:
    return require_table(_CONFIG, name)


def _get_str(section: str, key: str, default: str) -> str:
    return get_str(_section(section), key, default)


def _get_int(section: str, key: str, default: int) -> int:
    return get_int(_section(section), key, default)


def _get_bool(section: str, key: str, default: bool) -> bool:
    return get_bool(_section(section), key, default)


def _get_path_value(key: str, default: str) -> str | Path:
    return _PATH_OVERRIDES.get(key, _get_str("paths", key, default))


def _get_stage_value(key: str, default: str) -> str:
    return _STAGE_OVERRIDES.get(key, _get_str("stages", key, default))


def _get_graph_checker_value(key: str, default: str) -> str | Path | bool:
    return _GRAPH_CHECKER_OVERRIDES.get(
        key,
        _get_str("graph_checker", key, default),
    )


# ============================================================================
# Path configuration
# ============================================================================


def get_analysis_output_root() -> Path:
    return resolve_path(
        _get_path_value("analysis_output_root", "dist/pybastion"),
        relative_to_target=True,
    )


def get_inventories_root() -> Path:
    return resolve_path(
        _get_path_value("inventories_root", "inventory"),
        base=get_analysis_output_root(),
    )


def get_integration_output_dir() -> Path:
    return resolve_path(
        _get_path_value("integration_output_dir", "integration-output"),
        base=get_analysis_output_root(),
    )


def get_spec_split_output_dir() -> Path:
    return resolve_path(
        _get_path_value("spec_split_output_dir", "specs"),
        base=get_integration_output_dir(),
    )


# ============================================================================
# Stage configuration
# ============================================================================


def get_stage1_format() -> str:
    value = _get_stage_value("stage1_format", "pickle").strip().lower()
    if value not in {"pickle", "yaml"}:
        raise ValueError(
            f"Invalid stages.stage1_format: {value!r}; expected 'pickle' or 'yaml'"
        )
    return value


def get_stage_output(stage: int) -> Path:
    output_dir = get_integration_output_dir()

    match stage:
        case 1:
            default = (
                "stage1-ei-cfg.pkl"
                if get_stage1_format() == "pickle"
                else "stage1-ei-cfg.yaml"
            )
            return resolve_path(
                _get_stage_value("stage1_output", default),
                base=output_dir,
            )
        case 2:
            return resolve_path(
                _get_stage_value(
                    "stage2_output",
                    "stage2-feature-flow-cases.yaml",
                ),
                base=output_dir,
            )
        case 3:
            return resolve_path(
                _get_stage_value(
                    "stage3_output",
                    "stage3-integration-test-specs.yaml",
                ),
                base=output_dir,
            )
        case 4:
            return get_spec_split_output_dir()
        case _:
            raise ValueError(f"Unsupported stage: {stage}")


def get_stage_input(stage: int) -> Path:
    if stage not in {2, 3, 4}:
        raise ValueError(f"Stage must be 2, 3, or 4 (got {stage})")

    if stage in {2, 3}:
        return get_stage_output(1)

    return get_stage_output(3)


def get_stage2_marker_inventory_output() -> Path:
    return resolve_path(
        _get_stage_value(
            "stage2_marker_inventory_output",
            "stage2-feature-marker-inventory.yaml",
        ),
        base=get_integration_output_dir(),
    )


def get_stage2_branch_points_output() -> Path:
    return resolve_path(
        _get_stage_value(
            "stage2_branch_points_output",
            "stage2-feature-branch-points.yaml",
        ),
        base=get_integration_output_dir(),
    )


def get_stage2_converge_points_output() -> Path:
    return resolve_path(
        _get_stage_value(
            "stage2_converge_points_output",
            "stage2-feature-converge-points.yaml",
        ),
        base=get_integration_output_dir(),
    )


# ============================================================================
# Graph checker configuration
# ============================================================================


def graph_checker_enabled() -> bool:
    value = _GRAPH_CHECKER_OVERRIDES.get("enabled")
    if value is not None:
        return bool(value)

    return _get_bool("graph_checker", "enabled", False)


def get_graph_checker_script() -> Path:
    configured = _get_graph_checker_value(
        "script",
        "utils/check_inventory_graph.py",
    )
    return resolve_path(configured, relative_to_target=False)


def get_graph_checker_report() -> Path:
    configured = _get_graph_checker_value(
        "report",
        "inventory-graph-report.yaml",
    )
    return resolve_path(configured, base=get_integration_output_dir())


def get_graph_checker_summary() -> str:
    value = _get_graph_checker_value("summary", "l")
    return str(value)


# ============================================================================
# Output format configuration
# ============================================================================


def get_yaml_width() -> int:
    return _get_int("output_format", "yaml_width", 100)


def get_yaml_indent() -> int:
    return _get_int("output_format", "yaml_indent", 2)


def get_yaml_sort_keys() -> bool:
    return _get_bool("output_format", "yaml_sort_keys", False)


def include_metadata() -> bool:
    return _get_bool("output_format", "include_metadata", True)


def debug_output() -> bool:
    return _get_bool("output_format", "debug_output", False)


# ============================================================================
# Logging configuration
# ============================================================================


def get_verbosity() -> int:
    return _get_int("logging", "verbosity", 1)


def show_progress() -> bool:
    return _get_bool("logging", "show_progress", True)


# ============================================================================
# Utility functions
# ============================================================================


def ensure_output_dir() -> None:
    get_integration_output_dir().mkdir(parents=True, exist_ok=True)
    get_spec_split_output_dir().mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    errors: list[str] = []

    try:
        graph_format = get_stage1_format()
    except ValueError as exc:
        errors.append(str(exc))
        graph_format = "pickle"

    inventories_root = get_inventories_root()
    if not inventories_root.exists():
        errors.append(f"Inventories root does not exist: {inventories_root}")

    verbosity = get_verbosity()
    if verbosity not in {0, 1, 2}:
        errors.append(f"verbosity must be 0, 1, or 2 (got {verbosity})")

    yaml_width = get_yaml_width()
    if yaml_width <= 0:
        errors.append(f"yaml_width must be > 0 (got {yaml_width})")

    yaml_indent = get_yaml_indent()
    if yaml_indent <= 0:
        errors.append(f"yaml_indent must be > 0 (got {yaml_indent})")

    stage1_output = get_stage_output(1)
    if graph_format == "pickle" and stage1_output.suffix not in {
        ".pkl",
        ".pickle",
    }:
        errors.append(
            "stage1_format is pickle but stage1_output does not look like "
            f"a pickle file: {stage1_output}"
        )

    if graph_format == "yaml" and stage1_output.suffix not in {".yaml", ".yml"}:
        errors.append(
            "stage1_format is yaml but stage1_output does not look like "
            f"a YAML file: {stage1_output}"
        )

    stage2_output = get_stage_output(2)
    if stage2_output.suffix not in {".yaml", ".yml"}:
        errors.append(f"stage2_output does not look like a YAML file: {stage2_output}")

    stage3_output = get_stage_output(3)
    if stage3_output.suffix not in {".yaml", ".yml"}:
        errors.append(f"stage3_output does not look like a YAML file: {stage3_output}")

    if get_stage1_format() != "pickle":
        errors.append(
            "Stage 2 currently requires stages.stage1_format = 'pickle' "
            "because feature-flow tracing loads the EI CFG pickle."
        )

    return errors


def print_config_summary() -> None:
    print("Configuration Summary:")
    print(f"  Target root:          {get_target_root()}")
    print(f"  Analysis output root: {get_analysis_output_root()}")
    print(f"  Inventories root:     {get_inventories_root()}")
    print(f"  Output dir:           {get_integration_output_dir()}")
    print(f"  Split specs dir:      {get_spec_split_output_dir()}")
    print(f"  Stage 1 output:       {get_stage_output(1)}")
    print(f"  Stage 1 format:       {get_stage1_format()}")
    print(f"  Stage 2 output:       {get_stage_output(2)}")
    print(f"  Stage 2 inventory:    {get_stage2_marker_inventory_output()}")
    print(f"  Stage 2 branches:     {get_stage2_branch_points_output()}")
    print(f"  Stage 2 converges:    {get_stage2_converge_points_output()}")
    print(f"  Stage 3 output:       {get_stage_output(3)}")
    print(f"  Stage 4 output dir:   {get_stage_output(4)}")
    print(f"  Graph checker:        {graph_checker_enabled()}")
    print(f"  Graph checker script: {get_graph_checker_script()}")
    print(f"  Graph check report:   {get_graph_checker_report()}")
