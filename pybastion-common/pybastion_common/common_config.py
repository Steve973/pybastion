from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Python <3.11 requires tomli: pip install tomli") from exc


def load_toml_config(path: Path) -> dict[str, Any]:
    """Load a TOML config file and require a table at the root."""
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to parse config file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Config root must be a table")

    return payload


def select_config_table(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Select a named config table from a combined config file.

    If the requested table exists, it is returned.

    If the requested table does not exist, the original payload is returned. This
    allows the same loader to accept both of these shapes:

        [unit.paths]
        ...

    and:

        [paths]
        ...
    """
    table = payload.get(name)

    if table is None:
        return payload

    if not isinstance(table, dict):
        raise ValueError(f"Config key '{name}' must be a table")

    return table


def require_table(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a table value or an empty table when absent."""
    value = payload.get(key, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a table")

    return value


def reject_unknown_keys(payload: dict[str, Any], allowed: set[str], location: str) -> None:
    """Reject unknown config keys at a specific schema location."""
    unknown = sorted(set(payload) - allowed)

    if unknown:
        rendered = ", ".join(unknown)
        raise ValueError(f"Unsupported config key(s) in {location}: {rendered}")


def get_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)

    if isinstance(value, bool):
        return value

    raise ValueError(f"Config key '{key}' must be a boolean")


def get_str(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)

    if isinstance(value, str):
        return value

    raise ValueError(f"Config key '{key}' must be a string")


def get_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)

    if isinstance(value, int):
        return value

    raise ValueError(f"Config key '{key}' must be an integer")


def get_optional_positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)

    if value is None:
        return None

    if isinstance(value, int) and value > 0:
        return value

    raise ValueError(f"Config key '{key}' must be a positive integer when provided")
