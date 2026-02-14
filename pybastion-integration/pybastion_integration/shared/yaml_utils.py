"""
YAML serialization utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that doesn't use aliases."""
    def ignore_aliases(self, data: Any) -> bool:
        return True


def yaml_dump(data: Any) -> str:
    """
    Dump data to YAML string with consistent formatting.

    Args:
        data: Data to serialize

    Returns:
        YAML string
    """
    return yaml.dump(
        data,
        Dumper=_NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        width=100,
        allow_unicode=True,
        explicit_start=False,
        explicit_end=False,
    )


def yaml_load(path: Path) -> dict[str, Any]:
    """
    Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML data
    """
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
