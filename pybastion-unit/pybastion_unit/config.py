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
    config_path: Path
    project_root: Path
    package_root: Path
    stages_root: Path
    source_root: Path
    analysis_output_root: Path
    inspect_root: Path
    eis_root: Path
    inventory_root: Path
    logs_root: Path
    stage1_script: Path
    stage2_script: Path
    stage3_script: Path
    readiness_script: Path
    readiness_output_path: Path
    readiness_format: str
    readiness_grouping: str
    readiness_enabled: bool
    readiness_max_findings_per_type: int | None
    continue_on_error: bool
    clean_logs: bool
    clean_outputs: bool

    @property
    def unit_index_path(self) -> Path:
        return self.inspect_root / "unit-index.json"


# TOML schema. Keep this intentionally narrow. Do not preserve removed keys.
_ALLOWED_TOP_LEVEL_KEYS = {"paths", "scripts", "behavior", "readiness"}
_ALLOWED_PATH_KEYS = {
    "source_root",
    "analysis_output_root",
    "inspect_root",
    "eis_root",
    "inventory_root",
    "logs_root",
}
_ALLOWED_SCRIPT_KEYS = {"stage1", "stage2", "stage3", "readiness"}
_ALLOWED_BEHAVIOR_KEYS = {"continue_on_error", "clean_logs", "clean_outputs"}
_ALLOWED_READINESS_KEYS = {
    "enabled",
    "output",
    "format",
    "grouping",
    "include_info",
    "exclude_dirs",
    "max_findings_per_type",
}


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


def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        rendered = ", ".join(unknown)
        raise ValueError(f"Unsupported config key(s) in {location}: {rendered}")


def _get_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config key '{key}' must be a boolean")


def _get_str(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value
    raise ValueError(f"Config key '{key}' must be a string")


def _get_optional_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"Config key '{key}' must be a positive integer when provided")


def _resolve_child_path(raw: str | None, *, parent: Path, default_name: str) -> Path:
    value = raw if raw is not None else default_name
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (parent / path).resolve()


def _resolve_package_path(raw: str | None, *, package_root: Path, default_name: str) -> Path:
    value = raw if raw is not None else default_name
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (package_root / path).resolve()


def _default_source_root(project_root: Path) -> Path:
    candidate = project_root / "src"
    return candidate if candidate.exists() else project_root


def _resolve_source_root(raw: str | None, *, project_root: Path) -> Path:
    if raw is None:
        return _default_source_root(project_root).resolve()
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def load_analysis_config(project_root: Path, config_path: Path | None = None) -> AnalysisConfig:
    project_root = project_root.resolve()
    package_root = Path(__file__).resolve().parent
    stages_root = package_root / "stages"

    resolved_config_path = (config_path or (package_root / "unit_config.toml")).resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    payload = _load_toml(resolved_config_path)
    _reject_unknown_keys(payload, _ALLOWED_TOP_LEVEL_KEYS, "config root")

    paths = _require_table(payload, "paths")
    scripts = _require_table(payload, "scripts")
    behavior = _require_table(payload, "behavior")
    readiness = _require_table(payload, "readiness")

    _reject_unknown_keys(paths, _ALLOWED_PATH_KEYS, "[paths]")
    _reject_unknown_keys(scripts, _ALLOWED_SCRIPT_KEYS, "[scripts]")
    _reject_unknown_keys(behavior, _ALLOWED_BEHAVIOR_KEYS, "[behavior]")
    _reject_unknown_keys(readiness, _ALLOWED_READINESS_KEYS, "[readiness]")

    source_root = _resolve_source_root(paths.get("source_root"), project_root=project_root)
    analysis_output_root = _resolve_child_path(
        paths.get("analysis_output_root"),
        parent=project_root,
        default_name="dist/pybastion",
    )
    inspect_root = _resolve_child_path(
        paths.get("inspect_root"),
        parent=analysis_output_root,
        default_name="inspect",
    )
    eis_root = _resolve_child_path(
        paths.get("eis_root"),
        parent=analysis_output_root,
        default_name="eis",
    )
    inventory_root = _resolve_child_path(
        paths.get("inventory_root"),
        parent=analysis_output_root,
        default_name="inventory",
    )
    logs_root = _resolve_child_path(
        paths.get("logs_root"),
        parent=analysis_output_root,
        default_name="logs",
    )

    stage1_script = _resolve_package_path(
        scripts.get("stage1"),
        package_root=package_root,
        default_name="stages/stage1_inspect_units.py",
    )
    stage2_script = _resolve_package_path(
        scripts.get("stage2"),
        package_root=package_root,
        default_name="stages/stage2_enumerate_exec_items.py",
    )
    stage3_script = _resolve_package_path(
        scripts.get("stage3"),
        package_root=package_root,
        default_name="stages/stage3_enumerate_callables.py",
    )
    readiness_script = _resolve_package_path(
        scripts.get("readiness"),
        package_root=package_root,
        default_name="analyze_readiness.py",
    )

    readiness_output_path = _resolve_child_path(
        readiness.get("output"),
        parent=inspect_root,
        default_name="readiness.yaml",
    )

    return AnalysisConfig(
        config_path=resolved_config_path,
        project_root=project_root,
        package_root=package_root,
        stages_root=stages_root,
        source_root=source_root,
        analysis_output_root=analysis_output_root,
        inspect_root=inspect_root,
        eis_root=eis_root,
        inventory_root=inventory_root,
        logs_root=logs_root,
        stage1_script=stage1_script,
        stage2_script=stage2_script,
        stage3_script=stage3_script,
        readiness_script=readiness_script,
        readiness_output_path=readiness_output_path,
        readiness_format=_get_str(readiness, "format", "yaml"),
        readiness_grouping=_get_str(readiness, "grouping", "smt"),
        readiness_enabled=_get_bool(readiness, "enabled", False),
        readiness_max_findings_per_type=_get_optional_positive_int(readiness, "max_findings_per_type"),
        continue_on_error=_get_bool(behavior, "continue_on_error", True),
        clean_logs=_get_bool(behavior, "clean_logs", True),
        clean_outputs=_get_bool(behavior, "clean_outputs", True),
    )


def validate_analysis_config(config: AnalysisConfig, *, require_readiness: bool = False) -> None:
    if not config.project_root.exists():
        raise FileNotFoundError(f"Target project root not found: {config.project_root}")
    if not config.project_root.is_dir():
        raise NotADirectoryError(f"Target project root is not a directory: {config.project_root}")
    if not config.source_root.exists():
        raise FileNotFoundError(f"Source root not found: {config.source_root}")
    if not config.source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {config.source_root}")

    for label, script in (
            ("stage1", config.stage1_script),
            ("stage2", config.stage2_script),
            ("stage3", config.stage3_script),
    ):
        if not script.exists():
            raise FileNotFoundError(f"Required {label} script not found: {script}")

    if require_readiness and not config.readiness_script.exists():
        raise FileNotFoundError(f"Required readiness script not found: {config.readiness_script}")

    if config.readiness_format not in {"yaml", "yml", "json", "markdown", "md"}:
        raise ValueError(f"Unsupported readiness format: {config.readiness_format}")
