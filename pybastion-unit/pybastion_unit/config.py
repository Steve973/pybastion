#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Python <3.11 requires tomli: pip install tomli") from exc


@dataclass(slots=True)
class AnalysisConfig:
    project_root: Path
    package_root: Path
    stages_root: Path
    source_root: Path
    output_root: Path
    inspect_root: Path
    eis_root: Path
    inventory_root: Path
    logs_root: Path
    stage1_script: Path
    stage2_script: Path
    stage3_script: Path
    emit_legacy_inventory: bool
    continue_on_error: bool
    clean_logs: bool
    clean_outputs: bool
    run_stage1: bool
    run_stage2: bool
    run_stage3: bool

    @property
    def unit_index_path(self) -> Path:
        return self.inspect_root / "unit-index.json"

    @property
    def legacy_inventory_path(self) -> Path:
        return self.inspect_root / "callable-inventory.txt"


def _load_toml(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a table")
    return payload


def _require_table(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a table")
    return value


def _get_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config key '{key}' must be a boolean")


def _resolve_project_path(raw: str | None, *, project_root: Path, default: Path) -> Path:
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_package_path(raw: str | None, *, package_root: Path, default: Path) -> Path:
    if raw is None:
        return default.resolve()
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (package_root / path).resolve()


def load_analysis_config(project_root: Path, config_path: Path | None = None) -> AnalysisConfig:
    project_root = project_root.resolve()
    package_root = Path(__file__).resolve().parent
    stages_root = package_root / "stages"

    if config_path is None:
        config_path = package_root / "unit_config.toml"

    payload: dict[str, Any] = {}
    if config_path is not None:
        config_path = config_path.resolve()
        if config_path.exists():
            payload = _load_toml(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    paths = _require_table(payload, "paths")
    scripts = _require_table(payload, "scripts")
    run = _require_table(payload, "run")
    behavior = _require_table(payload, "behavior")

    default_source_root = project_root / "src" if (project_root / "src").exists() else project_root
    output_root = _resolve_project_path(paths.get("output_root"), project_root=project_root,
                                        default=(project_root / "dist" / "pybastion"))
    inspect_root = _resolve_project_path(paths.get("inspect_root"), project_root=project_root,
                                         default=(output_root / "inspect"))
    eis_root = _resolve_project_path(paths.get("eis_root"), project_root=project_root, default=(output_root / "eis"))
    inventory_root = _resolve_project_path(paths.get("inventory_root"), project_root=project_root,
                                           default=(output_root / "inventory"))
    logs_root = _resolve_project_path(paths.get("logs_root"), project_root=project_root, default=(output_root / "logs"))
    source_root = _resolve_project_path(paths.get("source_root"), project_root=project_root,
                                        default=default_source_root)

    stage1_script = _resolve_package_path(
        scripts.get("stage1"),
        package_root=package_root,
        default=(stages_root / "stage1_inspect_units.py"),
    )
    stage2_script = _resolve_package_path(
        scripts.get("stage2"),
        package_root=package_root,
        default=(stages_root / "stage2_enumerate_exec_items.py"),
    )
    stage3_script = _resolve_package_path(
        scripts.get("stage3"),
        package_root=package_root,
        default=(stages_root / "stege3_enumerate_callables.py"),
    )

    return AnalysisConfig(
        project_root=project_root,
        package_root=package_root,
        stages_root=stages_root,
        source_root=source_root,
        output_root=output_root,
        inspect_root=inspect_root,
        eis_root=eis_root,
        inventory_root=inventory_root,
        logs_root=logs_root,
        stage1_script=stage1_script,
        stage2_script=stage2_script,
        stage3_script=stage3_script,
        emit_legacy_inventory=_get_bool(behavior, "emit_legacy_inventory", True),
        continue_on_error=_get_bool(behavior, "continue_on_error", True),
        clean_logs=_get_bool(behavior, "clean_logs", True),
        clean_outputs=_get_bool(behavior, "clean_outputs", True),
        run_stage1=_get_bool(run, "stage1", True),
        run_stage2=_get_bool(run, "stage2", True),
        run_stage3=_get_bool(run, "stage3", True),
    )


def validate_analysis_config(config: AnalysisConfig) -> None:
    if not config.project_root.exists():
        raise FileNotFoundError(f"Target project root not found: {config.project_root}")
    if not config.project_root.is_dir():
        raise NotADirectoryError(f"Target project root is not a directory: {config.project_root}")
    if not config.source_root.exists():
        raise FileNotFoundError(f"Source root not found: {config.source_root}")
    if not config.source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {config.source_root}")

    for enabled, label, script in (
            (config.run_stage1, "stage1", config.stage1_script),
            (config.run_stage2, "stage2", config.stage2_script),
            (config.run_stage3, "stage3", config.stage3_script),
    ):
        if enabled and not script.exists():
            raise FileNotFoundError(f"Required {label} script not found: {script}")
