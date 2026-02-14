"""
JSON schema validation for integration flow artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


def validate_against_schema(data: dict[str, Any], schema_path: Path) -> tuple[bool, list[str]]:
    """
    Validate data against JSON schema.

    Args:
        data: Data to validate
        schema_path: Path to JSON schema file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Schema validation error: {e}"]
